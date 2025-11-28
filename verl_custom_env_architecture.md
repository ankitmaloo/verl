# VERL Custom Environment Architecture

Complete setup for: **Inference → Multi-turn Chat → Environment → Reward → Training**

This document provides the architecture and code for integrating a custom environment with VERL's training/inference pipeline.

---

## **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INFERENCE PHASE                                             │
│     ├─ Load prompts from dataset                                │
│     ├─ Generate responses using language model                  │
│     └─ Output: (prompt, response) pairs                         │
│                                                                  │
│  2. ENVIRONMENT INTERACTION PHASE                               │
│     ├─ Parse response to extract action                         │
│     ├─ Execute action in environment                            │
│     ├─ Multi-turn chat: repeat until done                       │
│     └─ Output: trajectory with states, actions, observations    │
│                                                                  │
│  3. REWARD COMPUTATION PHASE                                    │
│     ├─ Evaluate trajectory in environment                       │
│     ├─ Compute reward signal                                    │
│     └─ Output: (trajectory, reward) pairs                       │
│                                                                  │
│  4. TRAINING PHASE                                              │
│     ├─ Compute advantages using GAE                             │
│     ├─ Compute PPO loss                                         │
│     ├─ Update policy and value networks                         │
│     └─ Repeat for next batch                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## **1. ENVIRONMENT CLASS (Your Implementation)**

Create your custom environment as a separate class:

```python
# custom_environment.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseEnvironment(ABC):
    """Base class for custom environments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = None
        self.done = False
        self.step_count = 0
        self.max_steps = config.get('max_steps', 10)
    
    @abstractmethod
    def reset(self) -> str:
        """
        Reset environment and return initial state description.
        
        Returns:
            str: Text description of initial state
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: Text action from language model
            
        Returns:
            observation: Text description of new state
            reward: Immediate reward signal
            done: Whether episode is finished
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_state_description(self) -> str:
        """Get current state as text description"""
        pass
    
    def is_done(self) -> bool:
        """Check if episode is finished"""
        return self.done or self.step_count >= self.max_steps


class CustomGameEnvironment(BaseEnvironment):
    """Example: Custom game environment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.score = 0
        self.inventory = []
        self.position = (0, 0)
    
    def reset(self) -> str:
        self.score = 0
        self.inventory = []
        self.position = (0, 0)
        self.step_count = 0
        self.done = False
        
        return self.get_state_description()
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action. Parse action from text.
        
        Example actions:
        - "move north"
        - "pick up sword"
        - "attack enemy"
        """
        self.step_count += 1
        
        # Parse action from text
        action_lower = action.lower().strip()
        reward = 0.0
        info = {}
        
        # Example: movement actions
        if "move north" in action_lower:
            self.position = (self.position[0], self.position[1] + 1)
            reward = 0.1
        elif "move south" in action_lower:
            self.position = (self.position[0], self.position[1] - 1)
            reward = 0.1
        elif "move east" in action_lower:
            self.position = (self.position[0] + 1, self.position[1])
            reward = 0.1
        elif "move west" in action_lower:
            self.position = (self.position[0] - 1, self.position[1])
            reward = 0.1
        
        # Example: item actions
        elif "pick up" in action_lower:
            item = action_lower.replace("pick up", "").strip()
            self.inventory.append(item)
            reward = 0.5
        
        # Example: goal achievement
        elif "complete quest" in action_lower:
            self.score += 10
            reward = 10.0
            self.done = True
            info['success'] = True
        
        else:
            reward = -0.1  # Penalty for invalid action
            info['invalid_action'] = True
        
        # Check termination
        if self.step_count >= self.max_steps:
            self.done = True
        
        return self.get_state_description(), reward, self.done, info
    
    def get_state_description(self) -> str:
        """Convert state to text description"""
        desc = f"""
        Game State:
        - Position: {self.position}
        - Score: {self.score}
        - Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}
        - Steps remaining: {self.max_steps - self.step_count}
        """
        return desc.strip()
```

---

## **2. INFERENCE CODE (From VERL)**

Copy and adapt from `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`:

```python
# inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class InferenceEngine:
    """Wrapper for model inference"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model.eval()
    
    def generate(self, 
                 prompts: List[str],
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> List[str]:
        """
        Generate responses for batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generated responses
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        responses = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Remove prompt from response
        responses = [
            response[len(prompt):].strip()
            for response, prompt in zip(responses, prompts)
        ]
        
        return responses
    
    def compute_log_probs(self, 
                         prompts: List[str],
                         responses: List[str]) -> torch.Tensor:
        """
        Compute log probabilities of responses given prompts.
        
        Returns:
            Tensor of shape (batch_size,) with log probs
        """
        # Concatenate prompt and response
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Get log probs
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            logits = outputs.logits
        
        # Compute log probs for response tokens
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Extract log probs for actual tokens
        batch_size = len(prompts)
        response_log_probs = []
        
        for i in range(batch_size):
            prompt_len = len(self.tokenizer.encode(prompts[i]))
            response_tokens = inputs['input_ids'][i, prompt_len:]
            
            # Get log prob for each token
            token_log_probs = []
            for j, token_id in enumerate(response_tokens):
                if token_id != self.tokenizer.pad_token_id:
                    token_log_prob = log_probs[i, prompt_len + j - 1, token_id]
                    token_log_probs.append(token_log_prob)
            
            # Sum log probs
            response_log_probs.append(sum(token_log_probs))
        
        return torch.tensor(response_log_probs, device=self.device)
```

---

## **3. MULTI-TURN CHAT HARNESS**

Orchestrates inference + environment interaction:

```python
# multi_turn_harness.py

from typing import List, Dict, Any, Tuple
from custom_environment import BaseEnvironment
from inference import InferenceEngine

class MultiTurnChatHarness:
    """Manages multi-turn interaction between model and environment"""
    
    def __init__(self, 
                 inference_engine: InferenceEngine,
                 environment: BaseEnvironment,
                 max_turns: int = 5):
        self.inference_engine = inference_engine
        self.environment = environment
        self.max_turns = max_turns
    
    def run_episode(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Run single episode: inference → environment → repeat
        
        Args:
            initial_prompt: Starting prompt for the episode
            
        Returns:
            episode_data: Dictionary containing trajectory and metadata
        """
        # Reset environment
        initial_state = self.environment.reset()
        
        # Build initial context
        context = f"{initial_prompt}\n\nInitial state:\n{initial_state}"
        
        trajectory = {
            'prompts': [initial_prompt],
            'states': [initial_state],
            'actions': [],
            'observations': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }
        
        total_reward = 0.0
        
        # Multi-turn loop
        for turn in range(self.max_turns):
            if self.environment.is_done():
                break
            
            # Generate action
            action = self.inference_engine.generate(
                [context],
                max_new_tokens=128,
                temperature=0.7
            )[0]
            
            trajectory['actions'].append(action)
            
            # Execute in environment
            observation, reward, done, info = self.environment.step(action)
            
            trajectory['observations'].append(observation)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['infos'].append(info)
            
            total_reward += reward
            
            # Update context for next turn
            context += f"\n\nAction: {action}\n\nObservation:\n{observation}"
            
            if done:
                break
        
        trajectory['total_reward'] = total_reward
        trajectory['num_turns'] = len(trajectory['actions'])
        
        return trajectory
    
    def run_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Run batch of episodes.
        
        Args:
            prompts: List of initial prompts
            
        Returns:
            List of episode trajectories
        """
        trajectories = []
        for prompt in prompts:
            trajectory = self.run_episode(prompt)
            trajectories.append(trajectory)
        
        return trajectories
```

---

## **4. REWARD COMPUTATION**

Compute rewards from trajectories (from `verl/trainer/ppo/reward.py`):

```python
# reward_manager.py

from typing import List, Dict, Any
import torch
from verl.protocol import DataProto

class RewardManager:
    """Computes rewards from environment trajectories"""
    
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment
    
    def compute_trajectory_reward(self, trajectory: Dict[str, Any]) -> float:
        """
        Compute reward for entire trajectory.
        
        Args:
            trajectory: Episode trajectory from multi-turn harness
            
        Returns:
            Total reward for trajectory
        """
        # Method 1: Sum of step rewards
        total_reward = sum(trajectory['rewards'])
        
        # Method 2: Bonus for success
        if trajectory['infos'] and trajectory['infos'][-1].get('success', False):
            total_reward += 5.0
        
        # Method 3: Penalty for invalid actions
        invalid_count = sum(1 for info in trajectory['infos'] if info.get('invalid_action', False))
        total_reward -= invalid_count * 0.5
        
        # Method 4: Efficiency bonus (fewer steps is better)
        efficiency_bonus = (self.environment.max_steps - trajectory['num_turns']) * 0.1
        total_reward += efficiency_bonus
        
        return total_reward
    
    def compute_batch_rewards(self, trajectories: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute rewards for batch of trajectories.
        
        Returns:
            Tensor of shape (batch_size,) with rewards
        """
        rewards = []
        for trajectory in trajectories:
            reward = self.compute_trajectory_reward(trajectory)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        Interface compatible with VERL trainer.
        
        Args:
            data: DataProto batch from trainer
            
        Returns:
            Tensor of rewards
        """
        # Extract trajectories from data
        # This depends on your data format
        # For now, return dummy rewards
        batch_size = len(data.batch)
        return torch.ones(batch_size, dtype=torch.float32)
```

---

## **5. TRAINING CODE (From VERL)**

Core PPO training (from `verl/trainer/ppo/core_algos.py`):

```python
# ppo_training.py

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class PPOTrainer:
    """PPO algorithm implementation"""
    
    def __init__(self, 
                 policy_model,
                 value_model,
                 learning_rate: float = 1e-5,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        self.policy_model = policy_model
        self.value_model = value_model
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.lam = lam
        
        self.policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=learning_rate)
    
    def compute_gae_advantages(self,
                               rewards: torch.Tensor,
                               values: torch.Tensor,
                               dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Shape (batch_size, seq_len)
            values: Shape (batch_size, seq_len)
            dones: Shape (batch_size, seq_len)
            
        Returns:
            advantages: Shape (batch_size, seq_len)
            returns: Shape (batch_size, seq_len)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.gamma * self.lam * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_ppo_loss(self,
                        old_log_probs: torch.Tensor,
                        new_log_probs: torch.Tensor,
                        advantages: torch.Tensor,
                        response_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO policy loss.
        
        Args:
            old_log_probs: Log probs from old policy
            new_log_probs: Log probs from new policy
            advantages: Computed advantages
            response_mask: Mask for valid tokens
            
        Returns:
            loss: PPO loss
            metrics: Dictionary of loss components
        """
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # Apply mask
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item()
        }
        
        return policy_loss, metrics
    
    def compute_value_loss(self,
                          values: torch.Tensor,
                          returns: torch.Tensor,
                          response_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute value function loss.
        """
        value_loss = F.mse_loss(values, returns, reduction='none')
        value_loss = (value_loss * response_mask).sum() / response_mask.sum()
        
        metrics = {
            'value_loss': value_loss.item(),
            'value_mean': values.mean().item(),
            'return_mean': returns.mean().item()
        }
        
        return value_loss, metrics
    
    def update(self,
               batch: Dict[str, torch.Tensor],
               num_epochs: int = 4) -> Dict[str, float]:
        """
        Perform PPO update on batch.
        
        Args:
            batch: Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - responses: Response token IDs
                - rewards: Rewards
                - old_log_probs: Old policy log probs
                
        Returns:
            metrics: Training metrics
        """
        all_metrics = {}
        
        for epoch in range(num_epochs):
            # Forward pass through policy
            policy_outputs = self.policy_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            new_log_probs = policy_outputs['log_probs']
            
            # Forward pass through value model
            value_outputs = self.value_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            values = value_outputs['values']
            
            # Compute advantages
            advantages, returns = self.compute_gae_advantages(
                batch['rewards'],
                values,
                batch['dones']
            )
            
            # Compute losses
            policy_loss, policy_metrics = self.compute_ppo_loss(
                batch['old_log_probs'],
                new_log_probs,
                advantages,
                batch['response_mask']
            )
            
            value_loss, value_metrics = self.compute_value_loss(
                values,
                returns,
                batch['response_mask']
            )
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Collect metrics
            all_metrics.update({
                f'epoch_{epoch}_policy_loss': policy_loss.item(),
                f'epoch_{epoch}_value_loss': value_loss.item(),
                f'epoch_{epoch}_total_loss': total_loss.item(),
                **policy_metrics,
                **value_metrics
            })
        
        return all_metrics
```

---

## **6. MAIN TRAINING LOOP**

Orchestrates everything:

```python
# main_training.py

import torch
from custom_environment import CustomGameEnvironment
from inference import InferenceEngine
from multi_turn_harness import MultiTurnChatHarness
from reward_manager import RewardManager
from ppo_training import PPOTrainer

def main():
    # Configuration
    config = {
        'model_path': 'meta-llama/Llama-2-7b-hf',
        'batch_size': 32,
        'num_epochs': 10,
        'num_steps_per_epoch': 100,
        'max_turns': 5,
        'device': 'cuda'
    }
    
    # Initialize components
    print("Initializing inference engine...")
    inference_engine = InferenceEngine(config['model_path'], device=config['device'])
    
    print("Initializing environment...")
    env_config = {'max_steps': config['max_turns']}
    environment = CustomGameEnvironment(env_config)
    
    print("Initializing multi-turn harness...")
    harness = MultiTurnChatHarness(
        inference_engine,
        environment,
        max_turns=config['max_turns']
    )
    
    print("Initializing reward manager...")
    reward_manager = RewardManager(environment)
    
    print("Initializing PPO trainer...")
    # Load policy and value models
    policy_model = inference_engine.model  # Use same model as policy
    value_model = inference_engine.model   # In practice, use separate value model
    
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        learning_rate=1e-5,
        clip_ratio=0.2
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch + 1} ===")
        
        # Generate prompts (in practice, load from dataset)
        prompts = [
            "You are in a game. Explore and complete quests.",
            "Navigate the world and find treasures.",
        ] * (config['batch_size'] // 2)
        
        # Inference + Environment interaction
        print("Running multi-turn episodes...")
        trajectories = harness.run_batch(prompts)
        
        # Compute rewards
        print("Computing rewards...")
        rewards = reward_manager.compute_batch_rewards(trajectories)
        
        # Prepare batch for training
        batch = {
            'input_ids': torch.randn(config['batch_size'], 512).long(),  # Placeholder
            'attention_mask': torch.ones(config['batch_size'], 512),
            'responses': torch.randn(config['batch_size'], 256).long(),
            'rewards': rewards.unsqueeze(-1).expand(-1, 256),
            'dones': torch.zeros(config['batch_size'], 256),
            'old_log_probs': torch.randn(config['batch_size'], 256),
            'response_mask': torch.ones(config['batch_size'], 256)
        }
        
        # Training step
        print("Updating policy...")
        metrics = ppo_trainer.update(batch, num_epochs=4)
        
        # Log metrics
        print(f"Metrics: {metrics}")
        avg_reward = rewards.mean().item()
        print(f"Average reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()
```

---

## **7. DATA FLOW SUMMARY**

```
Dataset (Prompts)
    ↓
Inference Engine (Generate responses)
    ↓
Multi-Turn Harness (Execute in environment)
    ↓
Trajectories (States, actions, observations)
    ↓
Reward Manager (Compute rewards)
    ↓
PPO Trainer (Update policy)
    ↓
Updated Policy Model
```

---

## **Key Integration Points**

1. **Environment**: Implement `BaseEnvironment` with `reset()`, `step()`, `get_state_description()`
2. **Inference**: Use `InferenceEngine` to generate actions from prompts
3. **Multi-turn**: `MultiTurnChatHarness` orchestrates repeated inference + environment steps
4. **Rewards**: `RewardManager` computes scalar rewards from trajectories
5. **Training**: `PPOTrainer` updates policy using standard PPO algorithm

---

*Document created by Cascade, AI Assistant*

*Powered by Penguin Alpha model, created by Cognition*
