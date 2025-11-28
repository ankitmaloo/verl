"""
Adapter for VERL's RayPPOTrainer with custom environment and tool support.

This module wraps VERL's production-grade distributed trainer to work with:
- Custom environments
- Multi-turn inference with tool calling
- Custom reward functions
"""

import logging
from typing import Dict, Optional, Callable, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VERLRayPPOTrainer
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
    from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
except ImportError as e:
    raise ImportError(
        f"VERL not installed or import failed: {e}\n"
        "Make sure you're running from the VERL repository and VERL is importable."
    )

from .env import ConfigStore, BaseEnvironment, Trajectory, RolloutData
from .inference import MultiTurnInferenceEngine, BaseInferenceEngine

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM RL DATASET ADAPTER
# ============================================================================

class CustomEnvironmentDataset(Dataset):
    """
    Adapter dataset for custom environments.
    
    Converts environment interactions into the format expected by VERL's trainer.
    """
    
    def __init__(
        self,
        environment: BaseEnvironment,
        tokenizer,
        num_samples: int = 100,
        prompt_template: str = "Perform this task: {state}",
    ):
        """
        Initialize dataset.
        
        Args:
            environment: Custom environment instance
            tokenizer: HuggingFace tokenizer
            num_samples: Number of samples to generate
            prompt_template: Template for generating prompts from environment state
        """
        self.environment = environment
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.prompt_template = prompt_template
        
        # Pre-generate prompts
        self.prompts = []
        for i in range(num_samples):
            obs = self.environment.reset()
            prompt = self.prompt_template.format(state=obs)
            self.prompts.append(prompt)
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        prompt = self.prompts[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "prompt": prompt,
        }


# ============================================================================
# REWARD FUNCTION WRAPPER
# ============================================================================

class EnvironmentRewardWrapper:
    """
    Wraps custom environment reward function for VERL's trainer.
    
    VERL expects reward_fn(data: DataProto) -> Dict with rewards
    We adapt custom environment rewards to this interface.
    """
    
    def __init__(
        self,
        environment: BaseEnvironment,
        multi_turn_engine: Optional[MultiTurnInferenceEngine] = None,
        custom_reward_fn: Optional[Callable] = None,
    ):
        """
        Initialize reward wrapper.
        
        Args:
            environment: Custom environment
            multi_turn_engine: Multi-turn inference engine with tools
            custom_reward_fn: Optional custom reward computation function
        """
        self.environment = environment
        self.multi_turn_engine = multi_turn_engine
        self.custom_reward_fn = custom_reward_fn
    
    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        Compute rewards for batch.
        
        Args:
            data: DataProto batch from VERL
            
        Returns:
            Tensor of shape (batch_size,) with rewards
        """
        batch_size = len(data.batch["input_ids"])
        rewards = torch.zeros(batch_size)
        
        # Extract responses from data
        if "responses" in data.batch:
            responses = data.batch["responses"]
        else:
            # Generate responses if not present
            logger.warning("No responses in batch, generating...")
            responses = []
        
        # Compute reward for each sample
        for i in range(batch_size):
            if isinstance(responses, torch.Tensor):
                response_text = self.environment.tokenizer.decode(
                    responses[i],
                    skip_special_tokens=True,
                )
            else:
                response_text = responses[i]
            
            # Execute in environment
            try:
                obs, reward, done, info = self.environment.step(response_text)
                
                # Apply custom reward function if provided
                if self.custom_reward_fn:
                    reward = self.custom_reward_fn(
                        trajectory=None,
                        response=response_text,
                        reward=reward,
                        info=info,
                    )
            except Exception as e:
                logger.warning(f"Environment step failed: {e}")
                reward = -1.0
            
            rewards[i] = reward
        
        return rewards


# ============================================================================
# CUSTOM PPO TRAINER WRAPPER
# ============================================================================

class CustomRayPPOTrainer(VERLRayPPOTrainer):
    """
    Extended VERL RayPPOTrainer with custom environment support.
    
    Inherits from VERL's RayPPOTrainer and adapts it for:
    - Custom environment interaction
    - Multi-turn inference with tools
    - Custom reward computation
    """
    
    def __init__(
        self,
        config: ConfigStore,
        environment: BaseEnvironment,
        tokenizer,
        reward_fn: Optional[Callable] = None,
        multi_turn_engine: Optional[MultiTurnInferenceEngine] = None,
        role_worker_mapping: Optional[Dict[Role, WorkerType]] = None,
        resource_pool_manager=None,
        **kwargs,
    ):
        """
        Initialize custom trainer.
        
        Args:
            config: ConfigStore configuration
            environment: Custom environment instance
            tokenizer: HuggingFace tokenizer
            reward_fn: Optional custom reward function
            multi_turn_engine: Optional MultiTurnInferenceEngine for multi-turn
            role_worker_mapping: Role to worker mapping (auto-created if None)
            resource_pool_manager: Resource pool manager (auto-created if None)
            **kwargs: Additional arguments passed to VERL's RayPPOTrainer
        """
        self.custom_config = config
        self.environment = environment
        self.multi_turn_engine = multi_turn_engine
        
        # Create dataset
        train_dataset = CustomEnvironmentDataset(
            environment=environment,
            tokenizer=tokenizer,
            num_samples=config.training.batch_size * 10,
        )
        
        # Wrap reward function
        reward_wrapper = EnvironmentRewardWrapper(
            environment=environment,
            multi_turn_engine=multi_turn_engine,
            custom_reward_fn=reward_fn,
        )
        
        # TODO: Setup role_worker_mapping and resource_pool_manager if not provided
        # This requires understanding VERL's worker types and roles
        
        # Call parent init with VERL config format
        # Note: This requires converting ConfigStore to VERL's OmegaConf format
        verl_config = self._create_verl_config(config)
        
        super().__init__(
            config=verl_config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping or self._default_role_mapping(),
            resource_pool_manager=resource_pool_manager or self._default_resource_pool(),
            reward_fn=reward_wrapper,
            train_dataset=train_dataset,
            **kwargs,
        )
        
        logger.info("CustomRayPPOTrainer initialized with environment integration")
    
    @staticmethod
    def _create_verl_config(config: ConfigStore):
        """
        Convert our ConfigStore to VERL's expected config format (OmegaConf).
        
        TODO: Full implementation requires mapping all fields to VERL's config schema.
        For now, returns a minimal config.
        """
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("OmegaConf required for VERL trainer")
        
        # TODO: Create full OmegaConf config from ConfigStore
        # This is a placeholder - you need to map all VERL config sections
        verl_config = OmegaConf.create({
            "model": {
                "model_name": config.model.model_name,
                "dtype": config.model.dtype,
            },
            "algorithm": {
                "gamma": config.training.gamma,
                "lam": config.training.lam,
                "clip_ratio": config.training.clip_ratio,
            },
            "trainer": {
                "total_epochs": config.training.num_epochs,
                "device": config.model.device,
                "project_name": "verl-custom",
                "experiment_name": "custom-env",
            },
            "data": {
                "train_batch_size": config.training.batch_size,
            },
        })
        
        return verl_config
    
    @staticmethod
    def _default_role_mapping() -> Dict[Role, WorkerType]:
        """Create default role to worker mapping."""
        # TODO: Implement proper mapping based on VERL's architecture
        return {}
    
    @staticmethod
    def _default_resource_pool():
        """Create default resource pool manager."""
        # TODO: Implement proper resource pool creation
        return None
    
    def fit(self, num_epochs: Optional[int] = None):
        """
        Fit the model.
        
        Args:
            num_epochs: Override number of epochs
        """
        if num_epochs:
            self.config.trainer.total_epochs = num_epochs
        
        logger.info(f"Starting training with custom environment integration")
        super().fit()


# ============================================================================
# SIMPLIFIED TRAINER ENTRY POINT
# ============================================================================

def create_trainer(
    config: ConfigStore,
    environment: BaseEnvironment,
    tokenizer,
    reward_fn: Optional[Callable] = None,
    multi_turn_engine: Optional[MultiTurnInferenceEngine] = None,
    **kwargs,
) -> CustomRayPPOTrainer:
    """
    Factory function to create trainer.
    
    Args:
        config: Configuration object
        environment: Custom environment
        tokenizer: HuggingFace tokenizer
        reward_fn: Optional reward function
        multi_turn_engine: Optional multi-turn engine
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured CustomRayPPOTrainer instance
    """
    return CustomRayPPOTrainer(
        config=config,
        environment=environment,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        multi_turn_engine=multi_turn_engine,
        **kwargs,
    )


# ============================================================================
# HELPER: GET VERL TRAINER DIRECTLY
# ============================================================================

def get_verl_trainer_class():
    """
    Get VERL's RayPPOTrainer class.
    
    Returns:
        VERLRayPPOTrainer class
    """
    return VERLRayPPOTrainer


def get_verl_location():
    """Get location of VERL's trainer in filesystem."""
    return "/Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/ray_trainer.py"
