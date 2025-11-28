# VERL Code Writing Skill

A comprehensive skill for writing, modifying, and debugging VERL RL training code. This skill provides structured guidance for implementing custom components across the four main RL pipeline stages: Inference, Training, Environment, and Algorithm.

## Quick Start

### What This Skill Does

Given a task like:
- "Add multi-turn conversation support"
- "Create custom reward function for math problems"
- "Change from GAE to GRPO advantage estimation"
- "Write a curriculum learning sampler"

This skill will:
1. **Identify** which component(s) need modification
2. **Locate** exact source files and line numbers
3. **Generate** correct, idiomatic code
4. **Provide** required config changes
5. **Explain** how it integrates with existing code

### How to Use

**Option 1: Ask for a modification**
```
"I want to use GRPO advantage estimation with a custom reward function for code generation tasks"
```

**Option 2: Ask for debugging help**
```
"My custom reward function isn't being called. Debug where the issue is."
```

**Option 3: Ask for new component implementation**
```
"Implement a curriculum learning sampler that increases problem difficulty over training"
```

---

## Reference: The Four Components

### Component 1: INFERENCE (Generation/Rollout)

**Entry Point:** `ray_trainer.py:1055` → `actor_rollout_wg.generate_sequences(gen_batch_output)`

**Source Files:**
- Main orchestration: `verl/trainer/ppo/ray_trainer.py:1054-1061`
- SGLang: `verl/workers/rollout/sglang_rollout/sglang_rollout.py` (recommended)
- vLLM: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- HuggingFace: `verl/workers/rollout/hf_rollout.py`

**Key Config Parameters:**
```yaml
actor_rollout_ref:
  rollout:
    name: "sglang"                  # Engine: sglang, vllm, hf
    mode: "async"                   # async or sync
    n: 5                            # num_return_sequences
    temperature: 1.0
    top_p: 0.9
    max_response_length: 1024
    tensor_model_parallel_size: 2
```

**Common Modifications:**
- ✅ Change inference engine
- ✅ Add multi-turn conversation
- ✅ Enable tool/function calling
- ✅ Modify sampling parameters
- ✅ Add custom generation logic

**Code Template for Custom Rollout:**
```python
# File: my_rollout.py
from verl.workers.rollout.base import BaseRollout
from verl import DataProto
import torch

class CustomRollout(BaseRollout):
    def generate_sequences(self, batch: DataProto) -> DataProto:
        """
        Generate sequences for a batch of prompts.

        Args:
            batch: Input batch with prompts

        Returns:
            DataProto with generated responses + log probs
        """
        # 1. Prepare inputs
        prompts = batch.prompts

        # 2. Call your inference engine
        outputs = self.inference_engine(
            prompts,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_length=self.config.max_response_length,
        )

        # 3. Extract responses and log probabilities
        responses = outputs['responses']
        log_probs = outputs['log_probs']

        # 4. Construct output DataProto
        output_batch = batch.clone()
        output_batch.responses = responses
        output_batch.response_log_probs = log_probs

        return output_batch
```

---

### Component 2: TRAINING (Policy & Value Updates)

**Entry Points:**
- Policy: `ray_trainer.py:1130` → `compute_policy_loss()`
- Value: `ray_trainer.py:1170` → `critic_wg.step()`
- Actor step: `ray_trainer.py:1160` → `actor_wg.step()`

**Source Files:**
- Loss functions: `verl/trainer/ppo/core_algos.py:50-150`
- Actor: `verl/workers/actor/dp_actor.py` (default) or `megatron_actor.py`
- Critic: `verl/workers/critic/dp_critic.py` (default) or `megatron_critic.py`

**Key Config Parameters:**
```yaml
actor_rollout_ref:
  actor:
    strategy: "fsdp"                # fsdp, fsdp2, megatron
    optim:
      lr: 1e-6
      beta1: 0.9
      beta2: 0.999
      weight_decay: 0.0
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 32
    use_kl_loss: True
    kl_loss_coef: 0.001
```

**Common Modifications:**
- ✅ Create custom policy loss function
- ✅ Add custom advantage estimator
- ✅ Modify optimizer or learning rate
- ✅ Add regularization terms
- ✅ Change batch sizes for stability

**Code Template: Custom Policy Loss**
```python
# File: verl/trainer/ppo/core_algos.py (add to this file)
from verl.trainer.ppo.core_algos import register_policy_loss
import torch

@register_policy_loss("my_custom_ppo")
def my_custom_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config=None,
    rollout_log_probs=None,
) -> tuple[torch.Tensor, dict]:
    """
    Custom policy loss implementation.

    Args:
        old_log_probs: Log probs from reference model [batch, seq_len]
        log_probs: Log probs from actor model [batch, seq_len]
        advantages: Advantage estimates [batch, seq_len]
        response_mask: Valid token mask [batch, seq_len]
        loss_agg_mode: How to aggregate (token_level, sample_level)
        config: Actor configuration
        rollout_log_probs: Optional rollout log probs

    Returns:
        loss: Scalar tensor
        loss_info: Dict with loss components
    """
    # Compute probability ratio
    ratio = (log_probs - old_log_probs).exp()

    # PPO clipped objective
    epsilon = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)

    # Optional: Add entropy bonus
    entropy_bonus = -log_probs.exp() * log_probs

    # Combine losses
    total_loss = policy_loss - 0.01 * entropy_bonus

    # Apply mask
    total_loss = total_loss * response_mask

    # Aggregate
    if loss_agg_mode == "token_level":
        loss = total_loss.sum() / response_mask.sum()
    else:
        loss = total_loss.mean()

    return loss, {
        "policy_loss": policy_loss.mean().item(),
        "entropy_bonus": entropy_bonus.mean().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
    }
```

**Code Template: Custom Advantage Estimator**
```python
# File: verl/trainer/ppo/core_algos.py (add to this file)
from verl.trainer.ppo.core_algos import register_adv_est
import torch

@register_adv_est("my_advantage")
def my_advantage_estimator(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom advantage estimation.

    Args:
        rewards: Reward tensor [batch, seq_len]
        values: Value estimates [batch, seq_len]
        dones: Done mask [batch, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: Advantage estimates [batch, seq_len]
        returns: Return estimates [batch, seq_len]
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Compute GAE
    next_value = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]

        # TD error
        td_error = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]

        # GAE
        advantages[:, t] = td_error + gamma * gae_lambda * (1 - dones[:, t]) * advantages[:, t + 1]
        returns[:, t] = advantages[:, t] + values[:, t]

    return advantages, returns
```

---

### Component 3: ENVIRONMENT/DATASET

**Entry Point:** `main_ppo.py:329` → `create_rl_dataset(config.data.train_files, ...)`

**Source Files:**
- Dataset loader: `main_ppo.py:369-416`
- Default dataset: `verl/utils/dataset/rl_dataset.py:RLHFDataset`
- Multi-turn: `verl/utils/dataset/multiturn_sft_dataset.py`
- Curriculum: `verl/experimental/dataset/sampler.py:AbstractSampler`
- Dynamic gen: `verl/utils/dataset/dynamicgen_dataset.py:DynamicGenDataset`

**Key Config Parameters:**
```yaml
data:
  train_files: null                 # List of data file paths
  val_files: null
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 1024
  shuffle: true
  seed: 42
  custom_cls:
    path: null                      # Custom dataset class path
    name: null                      # Custom dataset class name
  sampler:
    class_path: null                # Custom sampler for curriculum
    class_name: null
```

**Common Modifications:**
- ✅ Create custom dataset class
- ✅ Change data format/tokenization
- ✅ Add curriculum learning sampler
- ✅ Support different file formats

**Code Template: Custom Dataset**
```python
# File: my_dataset.py
import json
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """Custom dataset for VERL training."""

    def __init__(self, data_files, tokenizer, processor=None, config=None, **kwargs):
        """
        Args:
            data_files: List of file paths
            tokenizer: HuggingFace tokenizer
            processor: Optional image processor for multimodal
            config: Data configuration
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or {}
        self.data = []

        # Load data from files
        for file_path in data_files:
            if file_path.endswith('.json') or file_path.endswith('.jsonl'):
                with open(file_path, 'r') as f:
                    if file_path.endswith('.jsonl'):
                        items = [json.loads(line) for line in f]
                    else:
                        items = json.load(f)
                    self.data.extend(items if isinstance(items, list) else [items])
            elif file_path.endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(file_path)
                self.data.extend(df.to_dict('records'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract fields
        prompt = item.get('prompt', item.get('instruction', ''))
        response = item.get('response', item.get('output', ''))

        # Tokenize
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.get('max_prompt_length', 512),
        )

        response_tokens = self.tokenizer.encode(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.get('max_response_length', 1024),
        )

        return {
            'input_ids': torch.tensor(prompt_tokens, dtype=torch.long),
            'response_ids': torch.tensor(response_tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(prompt_tokens), dtype=torch.long),
        }
```

**Code Template: Custom Curriculum Sampler**
```python
# File: my_sampler.py
import torch
import numpy as np
from verl.experimental.dataset.sampler import AbstractSampler

class CurriculumSampler(AbstractSampler):
    """Curriculum learning sampler."""

    def __init__(self, data_source, data_config, **kwargs):
        self.data_source = data_source
        self.config = data_config
        self.num_samples = len(data_source)

        # Initialize difficulty scores (uniform)
        self.difficulty_scores = np.ones(self.num_samples)
        self.training_step = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        """Sample indices based on curriculum."""
        # Normalize scores to probabilities
        probs = self.difficulty_scores / self.difficulty_scores.sum()

        # Sample with replacement based on difficulty
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            p=probs,
            replace=True
        )

        return iter(indices.tolist())

    def update(self, indices, metrics):
        """Update difficulty scores based on training metrics."""
        for idx in indices:
            loss = metrics.get(idx, {}).get('loss', 0)

            # Increase difficulty for easy samples (low loss)
            if loss < 0.5:
                self.difficulty_scores[idx] *= 1.05
            # Decrease difficulty for hard samples (high loss)
            elif loss > 2.0:
                self.difficulty_scores[idx] *= 0.95

        self.training_step += 1
```

---

### Component 4: ALGORITHM (Reward & Advantage)

#### Part A: Reward Function

**Entry Point:** `main_ppo.py:317` → `load_reward_manager(config, tokenizer, ...)`

**Source Files:**
- Loader: `verl/trainer/ppo/reward.py:118-186`
- Custom fn loader: `verl/trainer/ppo/reward.py:61-115`
- Compute reward: `verl/trainer/ppo/reward.py:189-208`
- Default scoring: `verl/utils/reward_score/__init__.py`
- Math scoring: `verl/utils/reward_score/math_reward.py`
- GSM8K scoring: `verl/utils/reward_score/gsm8k.py`
- Managers: `verl/workers/reward_manager/`

**Key Config Parameters:**
```yaml
reward_model:
  enable: false                     # Use learned reward model?
  reward_manager: "naive"           # naive, batch, dapo, prime, limited
  reward_format: null               # gsm8k, math, custom
  custom_reward_function:
    path: null                      # Path to custom reward fn
    name: null                      # Function name
    reward_kwargs: {}
  sandbox_fusion:
    url: null                       # Sandbox service for code execution
    max_concurrent: 64
    memory_limit_mb: 1024
```

**Common Modifications:**
- ✅ Create custom reward scoring function
- ✅ Use sandbox execution for code problems
- ✅ Change reward manager type
- ✅ Add reward transformation/scaling

**Code Template: Custom Reward Function**
```python
# File: my_reward.py
import torch
from verl import DataProto
from typing import Optional

def compute_reward(
    data: DataProto,
    tokenizer=None,
    **kwargs
) -> torch.Tensor:
    """
    Compute rewards for generated responses.

    Args:
        data: DataProto with batch information
        tokenizer: Optional tokenizer for post-processing
        **kwargs: Additional reward configuration

    Returns:
        torch.Tensor: Reward tensor of shape [batch_size]
    """
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size, device=data.device)

    # Access generated responses
    responses = data.responses  # List of strings

    # Access ground truth if available
    ground_truths = data.get('ground_truths', [None] * batch_size)

    for i in range(batch_size):
        response = responses[i]
        ground_truth = ground_truths[i] if ground_truths else None

        # Your custom scoring logic
        score = compute_single_reward(response, ground_truth, **kwargs)
        rewards[i] = score

    return rewards

def compute_single_reward(response: str, ground_truth: Optional[str], **kwargs) -> float:
    """Score a single response."""
    # Example: Check if response contains the correct answer
    if ground_truth is None:
        return 0.0

    # Extract answer from response
    response_answer = extract_answer(response)
    truth_answer = extract_answer(ground_truth)

    if response_answer == truth_answer:
        return 1.0
    else:
        return 0.0

def extract_answer(text: str) -> str:
    """Extract answer from generated text."""
    # Your extraction logic
    import re
    matches = re.findall(r'answer[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()
```

#### Part B: Advantage Estimator

See **Component 2: TRAINING** section above (Custom Advantage Estimator template).

---

## Task Guides

### Task 1: Add Multi-Turn Conversation Support

**Problem:** Your dataset has multi-turn conversations, but the model is treating each turn independently.

**Solution:**

1. **Switch to SGLang** (best for multi-turn):
```bash
actor_rollout_ref.rollout.name=sglang
```

2. **Update data format** to include conversation history:
```python
# my_dataset.py
def __getitem__(self, idx):
    item = self.data[idx]

    # Assume item['messages'] = [{"role": "user", "content": "..."}, ...]
    conversation = item['messages']

    # Format as conversation
    formatted_prompt = format_conversation(conversation[:-1])  # All but last
    last_response = conversation[-1]['content']

    return {
        'prompt': formatted_prompt,
        'response': last_response,
    }

def format_conversation(messages):
    """Format messages for model input."""
    text = ""
    for msg in messages:
        role = msg['role'].upper()
        text += f"{role}: {msg['content']}\n"
    return text
```

3. **Set chat template in config:**
```bash
data.chat_template=chatml  # or alpaca, qwen, etc.
```

4. **Enable interactions in SGLang:**
```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    interactions:
      - conversation
```

### Task 2: Create Custom Reward Function

**Problem:** Default rewards don't work for your task.

**Solution:**

1. **Create reward file** (`my_reward.py`):
```python
import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.ground_truths[i]

        # Your scoring logic
        if check_correctness(response, ground_truth):
            rewards[i] = 1.0
        else:
            rewards[i] = 0.0

    return rewards

def check_correctness(response, ground_truth):
    # Implement your task-specific logic
    return response.strip() == ground_truth.strip()
```

2. **Update config:**
```bash
reward_model.custom_reward_function.path=my_reward.py \
reward_model.custom_reward_function.name=compute_reward
```

3. **Run training:**
```bash
python3 -m verl.trainer.main_ppo ... \
    reward_model.custom_reward_function.path=my_reward.py
```

### Task 3: Change Advantage Estimator

**Problem:** GAE advantage estimation isn't working well for your problem.

**Solution:**

1. **Try GRPO** (group-based, good for ranking):
```bash
algorithm.adv_estimator=grpo
```

2. **Or RLOO** (leave-one-out, variance reduction):
```bash
algorithm.adv_estimator=rloo
```

3. **Or create custom** (see Component 2 template above):
```python
# In verl/trainer/ppo/core_algos.py
@register_adv_est("my_adv")
def my_advantage(...):
    # Your implementation
    return advantages, returns
```

```bash
algorithm.adv_estimator=my_adv
```

### Task 4: Implement Curriculum Learning

**Problem:** Model struggles on hard problems initially.

**Solution:**

1. **Create curriculum sampler** (`my_sampler.py`):
```python
from verl.experimental.dataset.sampler import AbstractSampler
import numpy as np

class DifficultySampler(AbstractSampler):
    def __init__(self, data_source, data_config, **kwargs):
        self.data_source = data_source
        self.difficulty = np.ones(len(data_source))

    def __iter__(self):
        probs = self.difficulty / self.difficulty.sum()
        indices = np.random.choice(len(self.data_source), size=len(self.data_source), p=probs)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def update(self, indices, metrics):
        for idx in indices:
            if metrics[idx]['loss'] > threshold:
                self.difficulty[idx] *= 1.1  # Increase difficulty
```

2. **Update config:**
```bash
data.sampler.class_path=my_sampler.py \
data.sampler.class_name=DifficultySampler \
data.dataloader_num_workers=0
```

### Task 5: Add Tool/Function Calling

**Problem:** Model should be able to call tools (calculator, search, etc.).

**Solution:**

1. **Define tools** (`my_tools.yaml`):
```yaml
tools:
  - name: "calculate"
    description: "Perform math operations"
    parameters:
      - name: "expression"
        type: "string"
        description: "Math expression to evaluate"
```

2. **Switch to SGLang** and enable tools:
```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.tools=my_tools.yaml
```

3. **SGLang handles tool parsing** automatically.

---

## Debug Checklist

### Inference (Generation) Not Working

- [ ] Check `actor_rollout_ref.rollout.name` is set correctly
- [ ] Verify model path exists: `actor_rollout_ref.model.path`
- [ ] Check GPU memory: generation requires significant VRAM
- [ ] Look at `sglang_rollout.py:generate_sequences()` output shape

### Reward Function Not Being Called

- [ ] Verify reward file path exists: `reward_model.custom_reward_function.path`
- [ ] Check function signature: `(data: DataProto, **kwargs) -> torch.Tensor`
- [ ] Add prints in reward function to verify it's called
- [ ] Check `ray_trainer.py:1080` to see where `self.reward_fn()` is called

### Training Loss Exploding

- [ ] Check reward scale: should be in reasonable range
- [ ] Reduce learning rate: `actor_rollout_ref.actor.optim.lr`
- [ ] Check advantage scale: should have mean ~0, std ~1
- [ ] Enable gradient clipping in optimizer
- [ ] Verify policy loss is computed correctly

### Out of Memory (OOM)

- [ ] Reduce `ppo_micro_batch_size_per_gpu`
- [ ] Enable FSDP: `actor_rollout_ref.actor.strategy=fsdp`
- [ ] Enable param offload: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
- [ ] Reduce `data.train_batch_size`

### Custom Component Not Loading

- [ ] Check file path is absolute or relative to working directory
- [ ] Verify class/function name matches config exactly
- [ ] Check inheritance: must inherit from correct base class
- [ ] Look for import errors in logs

---

## Code Patterns to Use

### Pattern 1: Access Data in DataProto
```python
def my_function(data: DataProto):
    # Batch information
    batch_size = len(data.batch)

    # Responses
    responses = data.responses

    # Ground truth
    ground_truths = data.ground_truths

    # Metadata
    meta = data.meta_info

    # Tensors
    input_ids = data.input_ids
    log_probs = data.log_probs
```

### Pattern 2: Return DataProto from Custom Component
```python
def my_generator(batch: DataProto) -> DataProto:
    # Clone or copy batch
    output = batch.clone()

    # Modify fields
    output.responses = generated_responses
    output.log_probs = log_probabilities
    output.meta_info['my_field'] = value

    return output
```

### Pattern 3: Register Custom Function
```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_loss")
def my_loss_fn(old_log_probs, log_probs, advantages, **kwargs):
    # Implementation
    return loss, loss_info

# Later, use by name:
config.algorithm.policy_loss_fn = "my_loss"
```

### Pattern 4: Use Config in Custom Code
```python
def my_function(data, config=None, **kwargs):
    if config:
        # Access config values
        learning_rate = config.get('learning_rate', 1e-6)
        use_feature = config.get('use_feature', False)

    # Also get from kwargs
    extra_param = kwargs.get('extra_param', default)
```

---

## Quick Reference: Config Hierarchy

```
Base YAML: verl/trainer/config/ppo_trainer.yaml
    ↓ (includes)
Actor/Critic/Rollout configs: verl/trainer/config/{actor,critic,rollout}/
    ↓ (merged)
CLI overrides: algorithm.adv_estimator=grpo ...
    ↓ (result)
Final config object passed to TaskRunner.run()
```

**To override any config:**
```bash
python3 -m verl.trainer.main_ppo \
    path.to.config=value \
    another.config=123
```

---

## File Location Quick Map

```
┌─ INFERENCE
├─ SGLang: verl/workers/rollout/sglang_rollout/sglang_rollout.py
├─ vLLM: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
└─ HF: verl/workers/rollout/hf_rollout.py

┌─ TRAINING
├─ Losses: verl/trainer/ppo/core_algos.py:50+
├─ Actor: verl/workers/actor/dp_actor.py
├─ Critic: verl/workers/critic/dp_critic.py
└─ Loop: verl/trainer/ppo/ray_trainer.py:968+

┌─ ENVIRONMENT
├─ Dataset: verl/utils/dataset/rl_dataset.py
├─ Sampler: verl/experimental/dataset/sampler.py
└─ Loader: main_ppo.py:369+

┌─ ALGORITHM
├─ Reward: verl/trainer/ppo/reward.py:61+
├─ Scoring: verl/utils/reward_score/
├─ Advantage: verl/trainer/ppo/core_algos.py:200+
└─ Managers: verl/workers/reward_manager/
```

---

## Testing Custom Components

### Test Custom Reward Function
```python
# test_reward.py
from my_reward import compute_reward
from verl import DataProto
import torch

# Create dummy data
data = DataProto.from_single_dict({
    'responses': ['answer: 42'] * 4,
    'ground_truths': ['42'] * 4,
})

rewards = compute_reward(data)
print(f"Rewards: {rewards}")
assert rewards.shape == (4,)
assert rewards.min() >= 0 and rewards.max() <= 1
```

### Test Custom Loss Function
```python
# test_loss.py
from verl.trainer.ppo.core_algos import get_policy_loss_fn
import torch

loss_fn = get_policy_loss_fn("my_custom_loss")

old_log_probs = torch.randn(4, 10)
log_probs = torch.randn(4, 10)
advantages = torch.randn(4, 10)
mask = torch.ones(4, 10)

loss, loss_info = loss_fn(old_log_probs, log_probs, advantages, mask, "token_level")
print(f"Loss: {loss.item()}")
assert loss.isfinite()
```

### Test Custom Advantage
```python
# test_advantage.py
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
import torch

adv_fn = get_adv_estimator_fn("my_advantage")

rewards = torch.randn(4, 10)
values = torch.randn(4, 10)
dones = torch.zeros(4, 10)

adv, ret = adv_fn(rewards, values, dones)
print(f"Advantages: mean={adv.mean()}, std={adv.std()}")
print(f"Returns: mean={ret.mean()}, std={ret.std()}")
```

---

## Common Gotchas & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| Custom fn not loaded | Path is relative | Use absolute path or full module path |
| Reward returns wrong shape | Function returns [batch] instead of [batch, seq_len] | Reshape in compute_reward |
| Training diverges | Advantage scale too large | Normalize advantages before using |
| Config not applied | Typo in config name | Use `--cfg job` to print actual config |
| Model outputs wrong | Generation params wrong | Check sampling params in rollout config |
| Multi-turn broken | Using wrong engine | Switch to SGLang: `rollout.name=sglang` |
| Dataset too slow | No multiprocessing | Set `dataloader_num_workers > 0` |
| CUDA OOM | Batch too large | Reduce batch size or enable offloading |

---

## Success Criteria for Custom Components

### Custom Reward Function ✅
- [ ] Returns `torch.Tensor` of shape `[batch_size]`
- [ ] Values in reasonable range (e.g., [0, 1] or [-1, 1])
- [ ] Non-NaN values
- [ ] Deterministic (same input = same output)
- [ ] Works with DataProto format

### Custom Loss Function ✅
- [ ] Returns `(loss_tensor, loss_info_dict)`
- [ ] Loss is scalar (shape `[]`)
- [ ] Decreases during training
- [ ] Info dict has meaningful statistics
- [ ] Respects response_mask

### Custom Dataset ✅
- [ ] Inherits from `torch.utils.data.Dataset`
- [ ] Implements `__len__()` and `__getitem__()`
- [ ] Returns dict with required keys
- [ ] Handles tokenization correctly
- [ ] Works with DataLoader

### Custom Sampler ✅
- [ ] Inherits from `AbstractSampler`
- [ ] Implements `__iter__()` and `__len__()`
- [ ] Returns indices in range [0, len-1]
- [ ] Has optional `update()` method
- [ ] Compatible with curriculum learning

---

## Next Steps

1. **Identify your use case** from the Task Guides above
2. **Choose the component** to modify (Inference/Training/Env/Algo)
3. **Use the code template** for that component
4. **Test in isolation** (see Testing section)
5. **Update config** with path to your custom component
6. **Run training** and monitor logs for issues
7. **Debug** using the Debug Checklist

For specific questions, refer to:
- **ARCHITECTURE_MAP.md** - Overall structure
- **EXECUTION_TRACE.md** - Exact execution flow with line numbers
- **MODIFICATION_GUIDE.md** - Detailed modification examples
- **This file (VERL_CODE_SKILL.md)** - Implementation patterns and quick reference
