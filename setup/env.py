"""
Environment Configuration and Base Classes for VERL Custom Training.

This module provides:
1. ConfigStore: Central configuration for all training components
2. BaseEnvironment: Abstract base class for custom environments
3. Supporting data structures for state/action management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    dtype: str = "bfloat16"
    device: str = "cuda"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # For local model path (if not using HF model)
    local_path: Optional[str] = None
    trust_remote_code: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    backend: str = "vllm"  # "vllm" or "sglang"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    multi_turn: bool = False  # Enable multi-turn chat
    max_turns: int = 5  # Maximum turns in multi-turn mode


@dataclass
class RewardConfig:
    """Reward computation configuration."""
    # TODO: Define how rewards are computed (e.g., from environment, LLM evaluator, etc.)
    # Example:
    # reward_source: str = "environment"  # "environment", "llm_evaluator", "hybrid"
    # scaling_factor: float = 1.0
    pass


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    # TODO: Add environment-specific parameters
    # Example:
    # env_type: str = "custom"
    # max_steps: int = 10
    # state_dim: int = 128
    # action_space_size: int = 256
    pass


@dataclass
class TrainingConfig:
    """Training configuration."""
    # PPO hyperparameters
    learning_rate: float = 1e-5
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 1.0
    
    # PPO clipping
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    cliprange_value: float = 0.2
    
    # Training loop
    num_epochs: int = 3
    num_train_iters_per_epoch: int = 100
    batch_size: int = 32
    accumulation_steps: int = 4
    
    # Advantage estimation
    adv_estimator: str = "gae"  # "gae", "grpo", "reinforce_plus_plus", etc.
    norm_adv: bool = True
    
    # KL penalty
    use_kl_in_reward: bool = False
    kl_ctrl_type: str = "fixed"  # "fixed" or "adaptive"
    kl_coef: float = 0.01
    target_kl: float = 0.01
    kl_horizon: int = 10000


@dataclass
class RayConfig:
    """Ray distributed training configuration."""
    use_ray: bool = True
    num_actors: int = 1
    num_critics: int = 1
    num_gpus_per_actor: float = 1.0
    num_gpus_per_critic: float = 1.0
    num_cpus_per_actor: float = 1.0
    num_cpus_per_critic: float = 1.0
    # Resource specification for colocating workers
    resource_pool_spec: Dict[str, List[int]] = field(default_factory=lambda: {"default": [1]})


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    output_dir: str = "./checkpoints"
    save_interval: int = 10  # Save every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints
    resume_path: Optional[str] = None  # Path to resume from


@dataclass
class ConfigStore:
    """Central configuration store for all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigStore":
        """Create ConfigStore from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key) and isinstance(value, dict):
                setattr(config, key, type(getattr(config, key)).from_dict(value))
            elif hasattr(config, key):
                setattr(config, key, value)
        return config


# ============================================================================
# ENVIRONMENT BASE CLASSES
# ============================================================================

class BaseEnvironment(ABC):
    """
    Abstract base class for custom environments.
    
    Implement this class to integrate your custom environment with VERL's training pipeline.
    The environment should:
    1. Accept text actions from the language model
    2. Update internal state based on actions
    3. Return text observations and scalar rewards
    4. Support multi-turn interactions
    
    Example:
        class MyGameEnvironment(BaseEnvironment):
            def __init__(self, config: EnvironmentConfig):
                super().__init__(config)
                # Initialize your environment
                
            def reset(self) -> str:
                # Reset and return initial observation
                pass
                
            def step(self, action: str) -> Tuple[str, float, bool, Dict]:
                # Execute action, return (observation, reward, done, info)
                pass
    """
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize environment.
        
        Args:
            config: Environment configuration object
        """
        self.config = config
        self.step_count = 0
        self.episode_reward = 0.0
        self._done = False
    
    @abstractmethod
    def reset(self) -> str:
        """
        Reset the environment to initial state.
        
        Returns:
            str: Text description of the initial state/observation
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Text action from the language model
            
        Returns:
            observation: Text description of the new state
            reward: Immediate scalar reward
            done: Whether the episode is finished
            info: Dictionary with additional information (e.g., metadata, success indicators)
        """
        pass
    
    @abstractmethod
    def get_state_description(self) -> str:
        """
        Get current state as text description.
        
        Used for multi-turn chat prompts to provide context to the model.
        
        Returns:
            str: Text description of current state
        """
        pass
    
    def get_prompt_prefix(self) -> str:
        """
        Get system prompt/instruction prefix for the language model.
        
        This can be overridden to provide task-specific instructions.
        
        Returns:
            str: System prompt text
        """
        # TODO: Implement based on your environment
        return "You are an AI agent in a task. Respond with your next action."
    
    def is_done(self) -> bool:
        """Check if episode is finished."""
        return self._done
    
    def get_episode_reward(self) -> float:
        """Get total reward accumulated in current episode."""
        return self.episode_reward
    
    def get_step_count(self) -> int:
        """Get number of steps taken in current episode."""
        return self.step_count


class SimpleGameEnvironment(BaseEnvironment):
    """
    Simple example environment for testing.
    
    This is a minimal implementation showing how to extend BaseEnvironment.
    Replace with your actual environment.
    """
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.inventory: List[str] = []
        self.position: Tuple[int, int] = (0, 0)
        self.score: float = 0.0
    
    def reset(self) -> str:
        """Reset environment."""
        self.inventory = []
        self.position = (0, 0)
        self.score = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        self._done = False
        
        return self.get_state_description()
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action."""
        self.step_count += 1
        reward = 0.0
        info = {}
        
        action_lower = action.lower().strip()
        
        # TODO: Implement your action parsing and execution logic
        # Example:
        if "move" in action_lower:
            reward = 0.1
        elif "take" in action_lower or "pick" in action_lower:
            reward = 0.5
        elif "complete" in action_lower:
            reward = 10.0
            self._done = True
            info['success'] = True
        else:
            reward = -0.1
            info['invalid'] = True
        
        # Check termination conditions
        # TODO: Add your termination logic
        # Example:
        # if self.step_count >= self.config.max_steps:
        #     self._done = True
        
        self.episode_reward += reward
        
        return self.get_state_description(), reward, self._done, info
    
    def get_state_description(self) -> str:
        """Get current state as text."""
        # TODO: Format your state as descriptive text for the model
        desc = f"""
        State:
        - Position: {self.position}
        - Score: {self.score}
        - Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}
        - Steps: {self.step_count}
        """
        return desc.strip()
    
    def get_prompt_prefix(self) -> str:
        """Get system prompt."""
        # TODO: Customize for your task
        return "You are in a game. Analyze the state and respond with your next action."


# ============================================================================
# HELPER STRUCTURES
# ============================================================================

@dataclass
class Trajectory:
    """
    Represents a single episode trajectory.
    
    Attributes:
        states: List of state observations
        actions: List of actions taken
        rewards: List of immediate rewards
        dones: List of done flags
        log_probs: Log probabilities of actions under rollout policy
        values: Value function estimates at each step
    """
    states: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self.states)
    
    def get_return(self, gamma: float = 0.99) -> float:
        """Compute discounted return."""
        total_return = 0.0
        for t in reversed(range(len(self.rewards))):
            total_return = self.rewards[t] + gamma * total_return
        return total_return


@dataclass
class RolloutData:
    """Data structure for rollout batch."""
    trajectories: List[Trajectory]
    prompts: List[str]
    tokens_ids: Optional[np.ndarray] = None  # Tokenized prompts
    response_tokens: Optional[np.ndarray] = None  # Response tokens
    rewards: Optional[np.ndarray] = None  # Scalar rewards per trajectory
    
    def __len__(self) -> int:
        return len(self.trajectories)


def create_default_config() -> ConfigStore:
    """Create a default configuration."""
    return ConfigStore()


def merge_configs(base: ConfigStore, overrides: Dict[str, Any]) -> ConfigStore:
    """Merge override dict into base config."""
    # TODO: Implement deep merge if needed
    for key, value in overrides.items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base
