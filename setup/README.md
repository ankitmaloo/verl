# VERL Custom Training Setup

A modular, extraction of VERL's core PPO training infrastructure for custom environments and reward functions.

## Overview

This package provides four integrated modules:

1. **env.py** - Configuration and environment base classes
2. **algo.py** - Core PPO algorithm implementations  
3. **inference.py** - VLLM and SGLang inference engines
4. **trainer.py** - Ray-based PPO trainer

## Quick Start

### 1. Define Your Environment

```python
from setup import BaseEnvironment, EnvironmentConfig

class MyEnvironment(BaseEnvironment):
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        # Initialize your environment
    
    def reset(self) -> str:
        """Reset and return initial observation."""
        return "Initial state description"
    
    def step(self, action: str) -> tuple:
        """Execute action and return (observation, reward, done, info)."""
        reward = 1.0  # Compute your reward
        return "New state", reward, False, {}
    
    def get_state_description(self) -> str:
        """Return current state as text for model context."""
        return "Current state description"
```

### 2. Configure Training

```python
from setup import ConfigStore

config = ConfigStore()

# Model
config.model.model_name = "meta-llama/Llama-2-7b-hf"
config.model.dtype = "bfloat16"

# Inference
config.inference.backend = "vllm"  # or "sglang"
config.inference.max_tokens = 256
config.inference.temperature = 0.7

# Training
config.training.learning_rate = 1e-5
config.training.num_epochs = 10
config.training.batch_size = 32
config.training.clip_ratio = 0.2

# Checkpointing
config.checkpoint.output_dir = "./checkpoints"
config.checkpoint.save_interval = 5
```

### 3. Define Reward Function

```python
def compute_reward(trajectory, environment):
    """Compute reward for a trajectory."""
    # Implement your reward logic
    total_reward = sum(trajectory.rewards)
    return total_reward
```

### 4. Train

```python
from setup import RayPPOTrainer

# Create environment
env = MyEnvironment(config.environment)

# Create trainer
trainer = RayPPOTrainer(
    config=config,
    environment=env,
    reward_fn=compute_reward,
    dataset=None,  # Optional: provide dataset
)

# Train
trainer.fit(num_epochs=10)
```

## File Structure

```
setup/
├── __init__.py           # Package exports
├── env.py                # Configuration and environment classes
├── algo.py               # PPO algorithm core functions
├── inference.py          # Inference engines (VLLM/SGLang)
├── trainer.py            # Ray PPO trainer
└── README.md             # This file
```

## Detailed Usage

### env.py - Configuration and Environments

**ConfigStore** - Central configuration object:

```python
config = ConfigStore()
config.model.model_name = "meta-llama/Llama-2-7b-hf"
config.training.learning_rate = 1e-5
# All config sections are optional - use defaults or override
```

**BaseEnvironment** - Abstract base class to implement:

```python
class MyEnv(BaseEnvironment):
    def reset(self) -> str:
        """Called at episode start."""
        pass
    
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Called for each action."""
        pass
    
    def get_state_description(self) -> str:
        """Called to provide context to model."""
        pass
    
    def get_prompt_prefix(self) -> str:
        """Optional: system prompt for model."""
        return "You are a helpful assistant."
```

**Trajectory** - Represents episode rollout:

```python
trajectory = Trajectory()
trajectory.states = ["state1", "state2", ...]
trajectory.actions = ["action1", "action2", ...]
trajectory.rewards = [1.0, 0.5, ...]
trajectory.dones = [False, False, ..., True]
```

### algo.py - PPO Algorithm Functions

**Advantage Estimation:**

```python
from setup import algo

# GAE (Generalized Advantage Estimation)
advantages, returns = algo.compute_gae_advantage_return(
    token_level_rewards=rewards,  # (batch, seq_len)
    values=values,                 # (batch, seq_len)
    response_mask=mask,            # (batch, seq_len)
    gamma=0.99,
    lam=0.95,
)

# GRPO (Group Relative Policy Optimization)
advantages, returns = algo.compute_grpo_advantage(
    token_level_rewards=rewards,
    response_mask=mask,
    index=group_indices,  # For grouping samples
)
```

**Policy and Value Loss:**

```python
# PPO policy loss with clipping
policy_loss, metrics = algo.compute_policy_loss(
    old_log_probs=old_lp,
    log_probs=new_lp,
    advantages=advantages,
    response_mask=mask,
    clip_ratio=0.2,
)

# Value function loss
value_loss, metrics = algo.compute_value_loss(
    vpreds=values,
    returns=returns,
    old_values=old_values,
    response_mask=mask,
    cliprange_value=0.2,
)
```

**KL Penalty:**

```python
# Apply KL penalty to rewards (for instruction-following)
penalized_rewards, kl_value = algo.apply_kl_penalty(
    token_level_rewards=rewards,
    log_probs=lp,
    ref_log_probs=ref_lp,
    response_mask=mask,
    kl_coef=0.01,
)

# Adaptive KL controller
kl_ctrl = algo.KLController(
    init_kl_coef=0.01,
    target_kl=0.01,
    adaptive=True,
)
kl_ctrl.update(current_kl=0.015, n_steps=1024)
```

### inference.py - Inference Engines

**VLLM Backend:**

```python
from setup import create_inference_engine

engine = create_inference_engine(
    model_path="meta-llama/Llama-2-7b-hf",
    backend="vllm",
    config={
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
    }
)

output = engine.generate(
    prompts=["Hello, how are you?"],
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
# output.sequences -> ["..."]
# output.tokens -> np.array(shape=(batch, seq_len))
```

**SGLang Backend:**

```python
engine = create_inference_engine(
    model_path="meta-llama/Llama-2-7b-hf",
    backend="sglang",  # Switch backend easily
)
```

**Multi-Turn Chat:**

```python
from setup import MultiTurnInferenceEngine

multi_turn = MultiTurnInferenceEngine(
    engine=engine,
    max_turns=5,
    system_prompt="You are a helpful assistant.",
)

# Single turn
multi_turn.start_conversation("Initial prompt")
response = multi_turn.step(max_tokens=256)

# Full episode with environment
responses = multi_turn.run_episode(
    environment=my_env,
    initial_prompt="Start playing the game.",
    max_turns=5,
)
```

**Batch Inference:**

```python
from setup import BatchInferenceManager

manager = BatchInferenceManager(engine, batch_size=32)
output = manager.generate_batch(
    prompts=large_prompt_list,
    max_tokens=256,
)
```

### trainer.py - Ray PPO Trainer

**Initialize Trainer:**

```python
from setup import RayPPOTrainer

trainer = RayPPOTrainer(
    config=config,
    environment=env,
    reward_fn=reward_function,
    dataset=optional_dataset,
)
```

**Training Loop:**

```python
# Train for N epochs
trainer.fit(num_epochs=10)

# Or specific configuration
trainer.fit(
    num_epochs=10,
    num_iters_per_epoch=100,
)
```

**Checkpointing:**

```python
# Manually save
trainer.save_checkpoint(Path("./checkpoints/epoch_5"))

# Load checkpoint
trainer.load_checkpoint(Path("./checkpoints/epoch_5"))

# Access metrics
for metric_dict in trainer.metrics_history:
    print(f"Epoch {metric_dict['epoch']}: {metric_dict['policy_loss']}")
```

**Cleanup:**

```python
trainer.cleanup()  # Release GPU memory
```

## Configuration Reference

### ModelConfig
- `model_name`: HuggingFace model ID
- `dtype`: Data type (bfloat16, float16, float32)
- `device`: cuda or cpu
- `tensor_parallel_size`: TP degree
- `local_path`: Local model path (alternative to model_name)
- `trust_remote_code`: Trust HF remote code

### InferenceConfig
- `backend`: "vllm" or "sglang"
- `max_tokens`: Maximum generation length
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter
- `multi_turn`: Enable multi-turn chat
- `max_turns`: Maximum conversation turns

### TrainingConfig
- `learning_rate`: Optimizer learning rate
- `gamma`: Discount factor for RL
- `lam`: GAE lambda parameter
- `clip_ratio`: PPO clip range
- `num_epochs`: Training epochs
- `batch_size`: Training batch size
- `adv_estimator`: "gae", "grpo", "reinforce_plus_plus", etc.
- `use_kl_in_reward`: Apply KL penalty to rewards
- `kl_coef`: Initial KL coefficient

### CheckpointConfig
- `output_dir`: Where to save checkpoints
- `save_interval`: Save every N epochs
- `keep_last_n`: Keep last N checkpoints

## TODO Items in Code

The skeleton includes several TODO placeholders for customization:

1. **env.py - EnvironmentConfig**
   - Add environment-specific parameters

2. **env.py - BaseEnvironment.get_prompt_prefix()**
   - Customize system prompt for your task

3. **env.py - SimpleGameEnvironment.step()**
   - Implement actual action parsing and execution

4. **inference.py - MultiTurnInferenceEngine.format_conversation()**
   - Implement proper chat template (ChatML, Llama2, etc.)

5. **trainer.py - RayPPOTrainer.rollout()**
   - Integrate with actual dataset loader
   - Implement multi-turn episode execution

6. **trainer.py - RayPPOTrainer.train_step()**
   - Add entropy loss if needed
   - Add gradient accumulation

## Example: Simple Text Game

```python
from setup import BaseEnvironment, ConfigStore, RayPPOTrainer

class TextGameEnv(BaseEnvironment):
    """Simple text-based game."""
    
    def reset(self) -> str:
        self.score = 0
        self.inventory = []
        return "Welcome to the game! You are in a room."
    
    def step(self, action: str) -> tuple:
        reward = 0.0
        
        if "take" in action.lower():
            self.inventory.append("item")
            reward = 1.0
        elif "attack" in action.lower():
            reward = 5.0
            self._done = True
        else:
            reward = -0.1
        
        self.score += reward
        return self.get_state_description(), reward, self._done, {}
    
    def get_state_description(self) -> str:
        return f"Score: {self.score}, Inventory: {self.inventory}"

# Train
config = ConfigStore()
env = TextGameEnv(config.environment)
trainer = RayPPOTrainer(config, env, reward_fn=lambda t: sum(t.rewards))
trainer.fit(num_epochs=5)
```

## Dependencies

Required:
- torch
- transformers
- numpy

Optional (depending on inference backend):
- vllm (for VLLM backend)
- sglang (for SGLang backend)

Optional (for distributed training):
- ray

## Notes

- All tensors are moved to the configured device automatically
- Tokenizer padding is set to eos_token if not defined
- Reference model is only loaded if `use_kl_in_reward=True`
- Multi-turn mode requires environment implementation
- Checkpoints include model weights, optimizer state, and metadata

## License

Same as VERL (Apache 2.0)
