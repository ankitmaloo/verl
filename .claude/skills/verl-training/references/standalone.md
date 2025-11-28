# Standalone Scripts Pattern

Simple Python files that call VERL components directly. For fast iteration.

## The 4 Files

```
algorithm.py    # Main script - run this
inference.py    # Calls VERL rollout workers
training.py     # Calls VERL actor/critic workers
env.py          # YOUR DATA SOURCE - modify this for different tasks
config.yaml     # Simple config
```

## algorithm.py - Main Loop

```python
#!/usr/bin/env python3
"""Main training orchestrator."""
import yaml
import torch
import ray
from verl import DataProto
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
from transformers import AutoTokenizer

# Import from other files
from env import get_batch
from inference import generate
from training import update_weights

def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=config.get('num_gpus', 8))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

    # Load reward function
    reward_fn = load_reward_manager(
        config=config,
        tokenizer=tokenizer,
        num_examine=0,
    )

    # Get advantage estimator
    adv_fn = get_adv_estimator_fn(config['algorithm']['advantage_estimator'])

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")

        for step in range(config['steps_per_epoch']):
            # 1. Get batch from environment (e.g., 10 prompts)
            batch = get_batch(size=config['batch_size'])
            prompts = batch['prompts']
            ground_truths = batch.get('ground_truths')

            # 2. Generate responses (10 prompts × n gens = 100 trajectories)
            outputs = generate(prompts, n=config['n_generations'])

            # 3. Compute rewards
            data = DataProto.from_single_dict({
                'responses': outputs['responses'],
                'prompts': prompts * config['n_generations'],
                'ground_truths': ground_truths * config['n_generations'] if ground_truths else None,
            })
            rewards, _ = reward_fn(data)

            # 4. Compute advantages
            advantages = adv_fn(
                rewards=rewards,
                values=None,  # or get from critic
                gamma=config['algorithm'].get('gamma', 0.99),
            )

            # 5. Update weights
            metrics = update_weights(
                log_probs=outputs['log_probs'],
                advantages=advantages,
            )

            # Log
            print(f"  Step {step + 1}: Loss={metrics['loss']:.4f}, "
                  f"Reward={rewards.mean().item():.4f}")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'actor_state': training.actor.state_dict(),
            }, f'checkpoint_epoch_{epoch + 1}.pt')

if __name__ == '__main__':
    main()
```

## inference.py - Generation

```python
"""Inference: calls VERL rollout workers."""
import yaml
import torch
from verl import DataProto

# Choose your engine
# Option 1: SGLang (best for multi-turn, tool calling)
from verl.workers.rollout.sglang_rollout.sglang_rollout import SglangRollout as RolloutWorker

# Option 2: vLLM (high throughput)
# from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import VllmRollout as RolloutWorker

# Option 3: HuggingFace (simple, local)
# from verl.workers.rollout.hf_rollout import HFRollout as RolloutWorker

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize rollout worker once (expensive)
rollout_config = config['inference']
rollout = RolloutWorker(rollout_config, role='actor')

def generate(prompts, n=1):
    """
    Generate n responses for each prompt.

    Args:
        prompts: List[str] - input prompts
        n: int - number of generations per prompt

    Returns:
        dict with 'responses', 'log_probs'
    """
    # Create batch
    batch = DataProto.from_single_dict({
        'prompts': prompts,
        'prompt_lengths': [len(p) for p in prompts],
    })

    # Repeat for n generations
    batch = batch.repeat(n, interleave=True)

    # Generate
    outputs = rollout.generate_sequences(batch)

    return {
        'responses': outputs.responses,
        'log_probs': outputs.log_probs,
        'lengths': outputs.response_lengths,
    }

# For multi-turn:
# Set in config.yaml: inference.multi_turn: true
# Pass prompts as: [{'role': 'user', 'content': '...'}]

# For tool calling:
# Set in config.yaml: inference.tools: path/to/tools.yaml
```

## training.py - Weight Updates

```python
"""Training: calls VERL actor/critic workers."""
import yaml
import torch
from verl.trainer.ppo.core_algos import get_policy_loss_fn

# Choose your strategy
# Option 1: FSDP (multi-GPU)
from verl.workers.actor.dp_actor import DataParallelPPOActor as ActorWorker
from verl.workers.critic.dp_critic import DataParallelPPOCritic as CriticWorker

# Option 2: Single GPU
# Just use DataParallelPPOActor but set strategy='single' in config

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize workers once (expensive)
actor_config = config['training']['actor']
critic_config = config['training']['critic']

actor = ActorWorker(actor_config, role='actor')
critic = CriticWorker(critic_config, role='critic')

# Get loss function
policy_loss_fn = get_policy_loss_fn(config['algorithm'].get('policy_loss', 'ppo'))

def update_weights(log_probs, advantages, old_log_probs=None):
    """
    Update actor and critic weights.

    Args:
        log_probs: torch.Tensor [batch, seq_len]
        advantages: torch.Tensor [batch, seq_len]
        old_log_probs: Optional[torch.Tensor] - if None, uses log_probs

    Returns:
        dict with metrics
    """
    if old_log_probs is None:
        old_log_probs = log_probs.detach()

    # Compute policy loss
    response_mask = torch.ones_like(log_probs)
    policy_loss, loss_info = policy_loss_fn(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode='token_level',
    )

    # Actor backward pass
    actor.backward(policy_loss)

    # Critic loss (optional)
    critic_loss = 0.0
    # if you have value predictions:
    # critic_loss = ((values - returns) ** 2).mean()
    # critic.backward(critic_loss)

    return {
        'loss': policy_loss.item(),
        'critic_loss': critic_loss if isinstance(critic_loss, float) else critic_loss.item(),
        **loss_info,
    }
```

## env.py - Data Source (MODIFY THIS)

```python
"""Environment: your data source. MODIFY THIS FILE for different tasks."""
import pandas as pd
import random

# ============================================
# Example 1: Simple Dataset
# ============================================

# Load data once
df = pd.read_parquet('data/train.parquet')
current_idx = 0

def get_batch(size=10):
    """Get next batch of prompts from dataset."""
    global current_idx

    # Get batch
    batch = df.iloc[current_idx:current_idx + size]
    current_idx = (current_idx + size) % len(df)

    return {
        'prompts': batch['question'].tolist(),
        'ground_truths': batch['answer'].tolist(),
    }

# ============================================
# Example 2: RL Game Environment
# ============================================

# class GameEnvironment:
#     """LLM playing a text-based game."""
#
#     def __init__(self):
#         self.states = []  # Track game states
#         self.reset()
#
#     def reset(self):
#         """Reset game to initial state."""
#         self.states = ["You are in a room. There is a door."] * 10
#
#     def get_batch(self, size=10):
#         """Get current game states as prompts."""
#         return {
#             'prompts': self.states[:size],
#             'ground_truths': None,  # No ground truth in RL
#         }
#
#     def step(self, actions):
#         """Take actions and update game states."""
#         for i, action in enumerate(actions):
#             # Update game state based on action
#             self.states[i] = self.apply_action(self.states[i], action)
#
#         # Return rewards
#         rewards = [self.compute_reward(s) for s in self.states]
#         return rewards

# ============================================
# Example 3: Multi-Turn Conversations
# ============================================

# class ConversationEnvironment:
#     """Multi-turn conversation dataset."""
#
#     def __init__(self):
#         # Load conversations
#         self.conversations = pd.read_json('data/conversations.json')
#         self.current_turn = 0
#
#     def get_batch(self, size=10):
#         """Get conversation history up to current turn."""
#         batch = self.conversations.iloc[:size]
#
#         prompts = []
#         for _, conv in batch.iterrows():
#             # Format as multi-turn
#             messages = conv['messages'][:self.current_turn]
#             prompts.append(messages)
#
#         return {
#             'prompts': prompts,  # List of message lists
#             'ground_truths': None,
#         }
#
#     def advance_turn(self):
#         """Move to next turn in conversation."""
#         self.current_turn += 1

# ============================================
# Example 4: Curriculum Learning
# ============================================

# def get_batch(size=10, difficulty='easy'):
#     """Get batch filtered by difficulty."""
#     # Filter by difficulty level
#     subset = df[df['difficulty'] == difficulty]
#
#     # Sample random
#     batch = subset.sample(n=size)
#
#     return {
#         'prompts': batch['question'].tolist(),
#         'ground_truths': batch['answer'].tolist(),
#     }
```

## config.yaml - Simple Config

```yaml
# Model
model_path: /path/to/model  # e.g., Qwen/Qwen-7B

# Hardware
num_gpus: 8

# Training
num_epochs: 10
steps_per_epoch: 100
batch_size: 10
n_generations: 10  # generations per prompt
save_every: 5  # save checkpoint every N epochs

# Inference
inference:
  engine: sglang  # sglang, vllm, or hf
  temperature: 0.7
  top_p: 0.9
  max_response_length: 1024
  multi_turn: false
  tools: null  # path/to/tools.yaml

# Training
training:
  actor:
    strategy: fsdp  # fsdp or single
    lr: 1e-6
    batch_size_per_gpu: 32
    use_kl_loss: true
    kl_loss_coef: 0.001

  critic:
    strategy: fsdp
    lr: 1e-5

# Algorithm
algorithm:
  advantage_estimator: grpo  # gae, grpo, rloo, etc.
  policy_loss: ppo  # ppo, grpo, reinforce
  gamma: 0.99

# Reward (optional)
reward_model:
  custom_reward_function:
    path: my_reward.py
    name: compute_reward
```

## How to Use

1. **Copy the 4 files** (algorithm.py, inference.py, training.py, env.py)
2. **Modify env.py** for your task (dataset, RL game, multi-turn)
3. **Edit config.yaml** with your settings
4. **Run**: `python algorithm.py`

## Common Modifications

### Use vLLM instead of SGLang
In `inference.py`:
```python
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import VllmRollout as RolloutWorker
```

### Single GPU instead of FSDP
In `config.yaml`:
```yaml
training:
  actor:
    strategy: single
```

### Custom reward function
Create `my_reward.py`:
```python
import torch

def compute_reward(data, **kwargs):
    rewards = torch.zeros(len(data.responses))
    for i, response in enumerate(data.responses):
        # Your scoring logic
        rewards[i] = score_response(response)
    return rewards
```

Set in `config.yaml`:
```yaml
reward_model:
  custom_reward_function:
    path: my_reward.py
    name: compute_reward
```

### Enable multi-turn
In `config.yaml`:
```yaml
inference:
  multi_turn: true
```

In `env.py`, return prompts as message lists:
```python
return {
    'prompts': [
        [
            {'role': 'user', 'content': 'First message'},
            {'role': 'assistant', 'content': 'Response'},
            {'role': 'user', 'content': 'Follow-up'},
        ]
    ],
}
```

### Enable tool calling
Create `tools.yaml`:
```yaml
tools:
  - name: calculator
    description: Add two numbers
    parameters:
      - name: a
        type: number
      - name: b
        type: number
```

In `config.yaml`:
```yaml
inference:
  tools: tools.yaml
```

## Quick Reference: VERL Components Used

**Inference:**
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py:SglangRollout`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:VllmRollout`
- `verl/workers/rollout/hf_rollout.py:HFRollout`

**Training:**
- `verl/workers/actor/dp_actor.py:DataParallelPPOActor`
- `verl/workers/critic/dp_critic.py:DataParallelPPOCritic`
- `verl/trainer/ppo/core_algos.py:get_policy_loss_fn()`

**Algorithm:**
- `verl/trainer/ppo/reward.py:load_reward_manager()`
- `verl/trainer/ppo/core_algos.py:get_adv_estimator_fn()`

**Data:**
- `verl.DataProto` - data structure for batches

## Debugging

**Test inference only:**
```python
from inference import generate
outputs = generate(['Test prompt'], n=5)
print(outputs['responses'])
```

**Test env only:**
```python
from env import get_batch
batch = get_batch(size=5)
print(batch['prompts'])
```

**Test reward only:**
```python
from verl.trainer.ppo.reward import load_reward_manager
from verl import DataProto

reward_fn = load_reward_manager(config, tokenizer, num_examine=0)
data = DataProto.from_single_dict({'responses': ['test'], 'ground_truths': ['test']})
rewards = reward_fn(data)
print(rewards)
```
