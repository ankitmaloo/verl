# VERL Training Skill

Complete guide for building RL training pipelines with VERL framework.

## Structure

```
verl-training/
├── SKILL.md                    # Main skill documentation (entry point)
├── references/
│   ├── inference.md           # How to call model generation (SGLang/vLLM/HF)
│   ├── training.md            # Policy loss & optimization
│   ├── environment.md         # Datasets & curriculum learning
│   └── algorithms.md          # Reward functions & advantage estimation
├── scripts/
│   └── create_reward_template.py  # Helper to generate reward templates
└── README.md                  # This file
```

## Getting Started

1. **Read SKILL.md first** - Overview of the four components
2. **Choose your task** - Select which component to modify
3. **Read relevant reference** - Deep dive into that component
4. **Use scripts** - Generate templates or test components
5. **Implement** - Add your custom code
6. **Test** - Verify in isolation before training

## The Four Components

### 1. Inference
How to generate responses using SGLang, vLLM, or HuggingFace.
- Multi-turn conversations
- Tool/function calling
- Sampling parameters
- Custom rollout implementations

**Reference:** `references/inference.md`

### 2. Training
How to setup policy and value function optimization.
- Custom policy loss functions
- Advantage estimators (GAE, GRPO, RLOO, etc.)
- Learning rate and optimizer config
- Distributed training (FSDP, Megatron)
- Stability management

**Reference:** `references/training.md`

### 3. Environment
How to load data and setup curriculum learning.
- Dataset loading (parquet, JSON, custom)
- Custom dataset classes
- Data tokenization
- Curriculum learning samplers
- Dynamic data generation

**Reference:** `references/environment.md`

### 4. Algorithm
How to implement reward functions and advantage estimation.
- Custom reward functions (math, code, QA, custom)
- Reward managers and scaling
- Advantage estimators
- Sandbox-based code execution rewards

**Reference:** `references/algorithms.md`

## Quick Examples

### Add Multi-Turn Conversation
```bash
actor_rollout_ref.rollout.name=sglang \
data.chat_template=chatml
```

### Create Custom Reward Function
```python
# my_reward.py
from verl import DataProto
import torch

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    rewards = torch.zeros(len(data.batch))
    for i, response in enumerate(data.responses):
        rewards[i] = score_response(response)
    return rewards
```

Config:
```bash
reward_model.custom_reward_function.path=my_reward.py \
reward_model.custom_reward_function.name=compute_reward
```

### Use GRPO Advantage Estimation
```bash
algorithm.adv_estimator=grpo
```

### Create Custom Dataset
```python
# my_dataset.py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, **kwargs):
        # Load your data
        pass

    def __getitem__(self, idx):
        return {
            'input_ids': ...,
            'response_ids': ...,
        }
```

Config:
```bash
data.custom_cls.path=my_dataset.py \
data.custom_cls.name=MyDataset
```

## Using Helper Scripts

### Generate Reward Function Template

```bash
python scripts/create_reward_template.py --task math --output my_reward.py
python scripts/create_reward_template.py --task code --output my_reward.py
python scripts/create_reward_template.py --task qa --output my_reward.py
```

## File Locations in VERL

All paths relative to VERL repo root:

```
Inference:
  verl/workers/rollout/sglang_rollout/sglang_rollout.py
  verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
  verl/workers/rollout/hf_rollout.py

Training:
  verl/trainer/ppo/core_algos.py
  verl/workers/actor/dp_actor.py
  verl/workers/critic/dp_critic.py
  verl/trainer/ppo/ray_trainer.py

Environment:
  verl/utils/dataset/rl_dataset.py
  verl/experimental/dataset/sampler.py
  verl/trainer/main_ppo.py

Algorithm:
  verl/trainer/ppo/reward.py
  verl/utils/reward_score/
  verl/workers/reward_manager/
```

## Common Workflows

### Workflow 1: Standard Training with Custom Rewards

1. Implement `compute_reward()` in `my_reward.py`
2. Set config:
   ```bash
   reward_model.custom_reward_function.path=my_reward.py
   ```
3. Run training with GRPO:
   ```bash
   python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
       reward_model.custom_reward_function.path=my_reward.py
   ```

### Workflow 2: Multi-Turn Training

1. Format dataset with conversation history
2. Set config:
   ```bash
   actor_rollout_ref.rollout.name=sglang \
   data.chat_template=chatml
   ```
3. Run training

### Workflow 3: Curriculum Learning

1. Implement custom sampler extending `AbstractSampler`
2. Set config:
   ```bash
   data.sampler.class_path=my_sampler.py \
   data.sampler.class_name=MyDifficultySampler \
   data.dataloader_num_workers=0
   ```
3. Run training (sampler updates difficulty during training)

### Workflow 4: Custom Policy Loss

1. Add function to `verl/trainer/ppo/core_algos.py` with `@register_policy_loss()`
2. Set config:
   ```bash
   algorithm.policy_loss_fn=my_custom_loss
   ```
3. Run training

## Debugging

### Verify Reward Function
```python
from my_reward import compute_reward
from verl import DataProto

# Create test data
data = DataProto.from_single_dict({
    'responses': ['answer: 42', 'answer: 100'],
    'ground_truths': [42, 100],
})

rewards = compute_reward(data)
print(rewards)  # Should be meaningful values
```

### Inspect Training Config
```bash
python3 -m verl.trainer.main_ppo --cfg job
```

### Monitor Metrics During Training
- Policy loss should decrease
- Value loss should decrease
- Reward mean should improve
- Ratio mean should stay near 1.0

## For Another Claude Instance

This skill is structured for another Claude instance to use:

1. **SKILL.md** - Quick reference and four component overview
2. **references/** - Deep technical documentation for each component
3. **scripts/** - Runnable helpers for common tasks
4. **This README** - Navigation and context

Start with SKILL.md, then dive into relevant reference as needed.

The skill covers:
- ✅ How to call inference (all engines)
- ✅ How to call training (losses, optimizers, strategies)
- ✅ How to setup environment (datasets, curriculum, sampling)
- ✅ How to use algorithms (rewards, advantage, custom implementations)
- ✅ Complete integration with VERL libraries

Based on comprehensive analysis of VERL codebase with exact line numbers and execution traces.
