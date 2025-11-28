# Quick Reference: How to Modify Each Component

## 1. Change Inference (Add Multi-Turn, Tool Calls, Different Engine)

### Use a Different Inference Engine (vLLM → SGLang → HF)

**Config change only:**
```bash
# In run_qwen3-8b.sh, change:
actor_rollout_ref.rollout.name=sglang    # from "vllm" to "sglang"

# Then run:
bash examples/grpo_trainer/run_qwen3-8b.sh
```

**Source code locations:**
- vLLM: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- SGLang: `verl/workers/rollout/sglang_rollout/sglang_rollout.py` (Recommended for multi-turn)
- HuggingFace: `verl/workers/rollout/hf_rollout.py`

---

### Enable Multi-Turn Conversations

**Option 1: Using SGLang (Recommended)**

In your config or script:
```bash
actor_rollout_ref.rollout.name=sglang \
data.chat_template=chatml  # or "alpaca", "qwen", etc.
```

**Option 2: Using Interactions Module**

```bash
actor_rollout_ref.rollout.interactions=[conversation_config]
```

Source: `verl/interactions/base.py` and `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~1800`

---

### Enable Tool/Function Calling

**Step 1: Create a tools YAML config** (e.g., `my_tools.yaml`):
```yaml
tools:
  - name: "calculator"
    description: "Add two numbers"
    parameters:
      - name: "a"
        type: "number"
      - name: "b"
        type: "number"
    function: "add_numbers"
```

**Step 2: Update your training config:**
```bash
actor_rollout_ref.rollout.tools=my_tools.yaml
```

**Source code:**
- Tool parsing: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~2000`
- Tool registry: `verl/tools/utils/tool_registry.py`

---

### Modify Generation Parameters (Temperature, Top-P, Max Tokens)

**Direct config change:**
```bash
actor_rollout_ref.rollout.temperature=0.7 \
actor_rollout_ref.rollout.top_p=0.9 \
actor_rollout_ref.rollout.max_response_length=1024
```

**Source code location:**
- For SGLang: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~400` (sampling params)
- For vLLM: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:~300`

---

## 2. Change Training (Loss Function, Optimizer, Learning Rate)

### Change Learning Rate and Optimizer Settings

**Config change:**
```bash
actor_rollout_ref.actor.optim.lr=5e-6 \
actor_rollout_ref.actor.optim.beta1=0.9 \
actor_rollout_ref.actor.optim.beta2=0.999 \
actor_rollout_ref.actor.optim.weight_decay=0.01
```

**Source code (if you need to modify optimizer type):**
- File: `verl/workers/actor/dp_actor.py:~80-120`
- Look for: `torch.optim.AdamW` initialization

---

### Add Custom Policy Loss Function

**Step 1: Add loss function to `core_algos.py`:**

```python
# File: verl/trainer/ppo/core_algos.py
# Add near line 200+ where other losses are defined

@register_policy_loss("my_custom_loss")
def my_custom_loss_fn(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config: Optional[ActorConfig] = None,
    rollout_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Custom policy loss function.

    Args:
        old_log_probs: Log probabilities from reference model
        log_probs: Log probabilities from actor model
        advantages: Advantage estimates
        response_mask: Mask for valid tokens
        loss_agg_mode: How to aggregate loss
        config: Actor config
        rollout_log_probs: Optional rollout log probs

    Returns:
        loss, loss_info_dict
    """
    # Your custom loss computation
    ratio = (log_probs - old_log_probs).exp()

    # Example: PPO with custom entropy bonus
    ppo_loss = -torch.min(
        ratio * advantages,
        ratio.clamp(1-0.2, 1+0.2) * advantages
    )

    # Add custom regularization
    custom_reg = torch.abs(log_probs).mean() * 0.01

    loss = ppo_loss.mean() + custom_reg

    loss_info = {
        "ppo_loss": ppo_loss.mean().item(),
        "custom_reg": custom_reg.item(),
    }

    return loss, loss_info
```

**Step 2: Use in config:**
```bash
algorithm.policy_loss_fn=my_custom_loss
```

**Step 3: Run training:**
```bash
python3 -m verl.trainer.main_ppo ... algorithm.policy_loss_fn=my_custom_loss
```

---

### Add Custom Advantage Estimator

**Step 1: Add to `core_algos.py`:**

```python
# File: verl/trainer/ppo/core_algos.py
# Add near line 400+ where other advantage estimators are defined

@register_adv_est("my_custom_advantage")
def my_custom_advantage_fn(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom advantage estimation.

    Args:
        rewards: Reward tensor [batch_size, seq_len]
        values: Value estimates [batch_size, seq_len]
        dones: Done mask [batch_size, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages, returns
    """
    batch_size, seq_len = rewards.shape

    # Your custom advantage logic
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    # Example: Simple TD advantage
    for t in range(seq_len - 1):
        td_error = rewards[:, t] + gamma * values[:, t+1] * (1 - dones[:, t]) - values[:, t]
        advantages[:, t] = td_error
        returns[:, t] = td_error + values[:, t]

    returns[:, -1] = rewards[:, -1]
    advantages[:, -1] = rewards[:, -1] - values[:, -1]

    return advantages, returns
```

**Step 2: Use in config:**
```bash
algorithm.adv_estimator=my_custom_advantage
```

---

### Change Distributed Training Strategy (FSDP → Megatron)

**Config change:**
```bash
actor_rollout_ref.actor.strategy=megatron \
actor_rollout_ref.critic.strategy=megatron
```

**Source code:** `verl/workers/megatron_workers.py`

---

## 3. Change Dataset/Environment

### Use a Different Data Source

**Config change:**
```bash
data.train_files=/path/to/my/data.parquet \
data.val_files=/path/to/my/val.parquet
```

**File format supported:**
- Parquet files (.parquet)
- JSON files (.json, .jsonl)

---

### Create Custom Dataset Class

**Step 1: Create custom dataset file** (e.g., `my_dataset.py`):

```python
# File: my_dataset.py
import torch
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, **kwargs):
        """
        Args:
            data_files: List of file paths
            tokenizer: HuggingFace tokenizer
            config: Data config object
        """
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

        # Load your data
        for file_path in data_files:
            with open(file_path, 'r') as f:
                # Parse your format
                items = json.load(f)
                self.data.extend(items)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(item['prompt'])

        # Tokenize response
        response_ids = self.tokenizer.encode(item['response'])

        # Optional: ground truth answer
        if 'answer' in item:
            # Prepare as expected by training loop
            ground_truth = item['answer']

        return {
            'input_ids': torch.tensor(prompt_ids),
            'labels': torch.tensor(response_ids),
            'attention_mask': torch.ones_like(response_ids),
            'ground_truth': ground_truth,
        }
```

**Step 2: Update config:**
```bash
data.custom_cls.path=my_dataset.py \
data.custom_cls.name=MyCustomDataset
```

**Step 3: Run training:**
```bash
python3 -m verl.trainer.main_ppo ... \
    data.custom_cls.path=my_dataset.py \
    data.custom_cls.name=MyCustomDataset
```

**Source code reference:** `verl/utils/dataset/rl_dataset.py:RLHFDataset` (example implementation)

---

### Use Curriculum Learning / Dynamic Sampling

**Step 1: Create sampler** (e.g., `my_sampler.py`):

```python
# File: my_sampler.py
from verl.experimental.dataset.sampler import AbstractSampler
import torch

class MyCurriculumSampler(AbstractSampler):
    def __init__(self, data_source, data_config, **kwargs):
        self.data_source = data_source
        self.config = data_config
        self.difficulty_scores = [0.5] * len(data_source)  # Initial scores

    def __iter__(self):
        # Sample based on curriculum (e.g., difficulty)
        indices = torch.multinomial(
            torch.tensor(self.difficulty_scores),
            len(self.data_source),
            replacement=True
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)

    def update(self, indices, metrics):
        # Update difficulty scores based on training performance
        for idx in indices:
            if metrics[idx]['loss'] > threshold:
                self.difficulty_scores[idx] *= 1.1  # Increase difficulty
```

**Step 2: Update config:**
```bash
data.sampler.class_path=my_sampler.py \
data.sampler.class_name=MyCurriculumSampler \
data.dataloader_num_workers=0
```

---

## 4. Change Reward Function

### Use Pre-built Reward Scoring (Math, GSM8K)

**Config change:**
```bash
reward_model.reward_format=gsm8k
```

**Available formats:**
- `gsm8k`: For GSM8K dataset
- `math`: For general math problems
- Custom path to scoring function

---

### Write Custom Reward Function

**Step 1: Create reward file** (e.g., `my_reward.py`):

```python
# File: my_reward.py
import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    """
    Compute rewards for a batch of generated responses.

    Args:
        data: DataProto with batch info
        **kwargs: Additional arguments from reward_kwargs

    Returns:
        torch.Tensor: Reward tensor of shape [batch_size]
    """
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    # Get responses and ground truths from data
    responses = data.responses  # Generated text
    ground_truths = data.ground_truths  # Expected answers

    for i in range(batch_size):
        response = responses[i]
        ground_truth = ground_truths[i]

        # Your custom scoring logic
        if is_correct_answer(response, ground_truth):
            rewards[i] = 1.0
        else:
            # Partial credit based on similarity
            similarity = compute_similarity(response, ground_truth)
            rewards[i] = similarity

    return rewards

def is_correct_answer(response: str, ground_truth: str) -> bool:
    """Check if response is correct."""
    # Extract numeric answer from response
    try:
        response_ans = extract_answer(response)
        truth_ans = extract_answer(ground_truth)
        return abs(response_ans - truth_ans) < 1e-6
    except:
        return False

def extract_answer(text: str) -> float:
    """Extract numeric answer from text."""
    import re
    # Look for patterns like "The answer is 42" or just "42"
    matches = re.findall(r'\d+\.?\d*', text)
    if matches:
        return float(matches[-1])
    return 0.0

def compute_similarity(response: str, ground_truth: str) -> float:
    """Compute similarity between response and ground truth."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, response, ground_truth).ratio()
```

**Step 2: Update config:**
```bash
reward_model.custom_reward_function.path=my_reward.py \
reward_model.custom_reward_function.name=compute_reward \
reward_model.custom_reward_function.reward_kwargs='{"key": "value"}'
```

**Step 3: Run training:**
```bash
python3 -m verl.trainer.main_ppo ... \
    reward_model.custom_reward_function.path=my_reward.py \
    reward_model.custom_reward_function.name=compute_reward
```

---

### Use Sandbox Execution for Code Problems

**Config change:**
```bash
reward_model.sandbox_fusion.url=http://localhost:8000 \
reward_model.sandbox_fusion.max_concurrent=64 \
reward_model.sandbox_fusion.memory_limit_mb=1024
```

**What this does:**
- Sends generated code to sandbox service
- Executes code with test cases
- Returns pass/fail based on test results

**Source:** `verl/utils/reward_score/sandbox_fusion/`

---

### Change Reward Manager Type

**Config options:**
```bash
reward_model.reward_manager=naive       # Basic reward computation
reward_model.reward_manager=batch       # Batch processing of rewards
reward_model.reward_manager=dapo        # DAPO-specific reward handling
reward_model.reward_manager=prime       # PRIME algorithm rewards
reward_model.reward_manager=limited     # Rate-limited reward computation
```

**Source files:**
- Naive: `verl/workers/reward_manager/naive.py`
- Batch: `verl/workers/reward_manager/batch.py`
- DAPO: `verl/workers/reward_manager/dapo.py`
- PRIME: `verl/workers/reward_manager/prime.py`
- Limited: `verl/experimental/reward/reward_loop/limited.py`

---

## 5. Change Advantage Estimator

### Switch Between Pre-built Estimators

**Config change:**
```bash
# GAE (General Advantage Estimation)
algorithm.adv_estimator=gae

# GRPO (Group Relative Policy Optimization) - YOUR CURRENT CHOICE
algorithm.adv_estimator=grpo

# REINFORCE with baseline
algorithm.adv_estimator=reinforce_plus_plus

# REINFORCE++ with value function
algorithm.adv_estimator=reinforce_plus_plus_baseline

# REMAX (Relative Expectation Maximization)
algorithm.adv_estimator=remax

# RLOO (Leave-One-Out)
algorithm.adv_estimator=rloo

# Others
algorithm.adv_estimator=opo
algorithm.adv_estimator=gpg
algorithm.adv_estimator=grpo_vectorized
algorithm.adv_estimator=rloo_vectorized
```

**Source code:** `verl/trainer/ppo/core_algos.py:~400+`

**How to compare them:**
1. Run training with each estimator
2. Look at advantage statistics in logs
3. Check training stability and convergence

---

### Create Custom Advantage Estimator

See **Section 2: Add Custom Advantage Estimator** above.

---

## Complete Example: Change Everything

Let's say you want to:
1. Use SGLang instead of vLLM
2. Add multi-turn conversation
3. Use custom reward function
4. Use REINFORCE++ advantage estimation

**Create `my_reward.py`:**
```python
import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    batch_size = len(data.batch)
    rewards = torch.ones(batch_size) * 0.5  # Placeholder
    # Add your logic
    return rewards
```

**Run training:**
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    actor_rollout_ref.rollout.name=sglang \
    data.chat_template=chatml \
    reward_model.custom_reward_function.path=my_reward.py \
    reward_model.custom_reward_function.name=compute_reward \
    data.train_files=$HOME/data/mydata/train.parquet \
    data.val_files=$HOME/data/mydata/test.parquet
```

---

## How to Debug Each Component

### Debug Inference (Generation)

Add prints to: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:generate_sequences()`

```python
def generate_sequences(self, batch):
    print(f"Input batch size: {len(batch.batch)}")
    print(f"Prompt lengths: {batch.prompt_len}")
    # ... generation code ...
    print(f"Output tokens: {output[:, :10]}")  # First 10 tokens
    return output
```

### Debug Reward

Add prints to: `verl/trainer/ppo/reward.py:compute_reward()`

```python
def compute_reward(data, reward_fn):
    reward_tensor = reward_fn(data, return_dict=True)
    print(f"Reward shape: {reward_tensor.shape}")
    print(f"Reward mean: {reward_tensor.mean()}")
    print(f"Reward std: {reward_tensor.std()}")
    return reward_tensor
```

### Debug Advantages

Add prints to: `verl/trainer/ppo/core_algos.py` advantage function

```python
@register_adv_est("grpo")
def grpo_advantage(rewards, values, ...):
    print(f"Rewards: mean={rewards.mean()}, std={rewards.std()}")
    print(f"Values: mean={values.mean()}, std={values.std()}")
    advantages = ...
    print(f"Advantages: mean={advantages.mean()}, std={advantages.std()}")
    return advantages
```

### Debug Training Loop

Add prints to: `verl/trainer/ppo/ray_trainer.py:fit()`

```python
for batch_dict in self.train_dataloader:
    print(f"Batch size: {len(batch_dict['input_ids'])}")
    gen_batch_output = self.actor_rollout_wg.generate_sequences(...)
    print(f"Generated: {gen_batch_output.responses[:1]}")  # First response
    reward_tensor = self.reward_fn(gen_batch_output)
    print(f"Rewards: {reward_tensor.mean()}")
    # ... continue training ...
```

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **Custom reward function not loading** | Check path is absolute, function signature matches `(data: DataProto, **kwargs)` → `torch.Tensor` |
| **Multi-turn not working** | Ensure `actor_rollout_ref.rollout.name=sglang` and data has conversation format |
| **Training loss exploding** | Reduce learning rate, add gradient clipping, check reward scaling |
| **Out of memory (OOM)** | Reduce `ppo_micro_batch_size_per_gpu`, enable `fsdp_config.param_offload=True` |
| **Slow training** | Enable `actor_rollout_ref.rollout.mode=async`, increase `tensor_model_parallel_size` |
| **Poor convergence** | Try different `adv_estimator` or `policy_loss_fn`, tune KL coefficient |

---

## Testing Your Changes

### Test custom reward function in isolation:

```python
# test_reward.py
import sys
sys.path.insert(0, '/path/to/my_reward.py')
from my_reward import compute_reward
from verl import DataProto
import torch

# Create dummy data
data = DataProto({
    'responses': ['The answer is 42'] * 4,
    'ground_truths': ['42'] * 4,
})

# Test reward function
rewards = compute_reward(data)
print(f"Rewards: {rewards}")  # Should be close to [1, 1, 1, 1]

# Run: python test_reward.py
```

### Test custom advantage estimator:

```python
# test_advantage.py
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
import torch

adv_fn = get_adv_estimator_fn("my_custom_advantage")

rewards = torch.randn(4, 10)  # batch_size=4, seq_len=10
values = torch.randn(4, 10)
dones = torch.zeros(4, 10)

advantages, returns = adv_fn(rewards, values, dones)
print(f"Advantages shape: {advantages.shape}")
print(f"Returns shape: {returns.shape}")

# Run: python test_advantage.py
```

---

## Useful Config Inspection Commands

```bash
# Print entire config
python3 -m verl.trainer.main_ppo --cfg job

# Print config with resolve (evaluate all references)
python3 -m verl.trainer.main_ppo --cfg job --resolve

# Print specific config section
python3 -m verl.trainer.main_ppo algorithm --cfg job

# Dry run (load config, don't train)
# Add a simple return at start of RayPPOTrainer.fit()
```
