# Algorithms: Reward & Advantage Functions

How to implement custom reward functions and advantage estimation strategies.

## Table of Contents

1. Reward Functions
2. Creating Custom Rewards
3. Reward Managers
4. Advantage Estimators
5. Sandbox-Based Rewards
6. Integration in Training Loop

## Reward Functions

Rewards signal to the model whether generated responses are good or bad.

### Built-in Reward Formats

VERL provides pre-built scoring functions for common tasks:

**Math Problems (GSM8K)**
```bash
reward_model.reward_format=gsm8k
```
Extracts numeric answer and compares with ground truth.

**General Math**
```bash
reward_model.reward_format=math
```
Flexible math problem scoring with pattern matching.

**Custom Format**
```bash
# No built-in format, use custom reward function
reward_model.custom_reward_function.path=my_reward.py
```

**Location:** `verl/utils/reward_score/` contains scoring implementations.

### Reward Function Signature

All reward functions must follow this signature:

```python
def compute_reward(
    data: DataProto,
    **kwargs
) -> torch.Tensor:
    """
    Args:
        data: DataProto with batch information
        **kwargs: Additional config from reward_kwargs

    Returns:
        torch.Tensor: Shape [batch_size], values typically in [0, 1] or [-1, 1]
    """
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    # Your scoring logic
    for i in range(batch_size):
        response = data.responses[i]
        rewards[i] = score_response(response)

    return rewards
```

### DataProto Fields Available

```python
data.responses       # List of generated strings
data.batch           # Original batch data
data.prompts         # Original prompts
data.ground_truths   # Ground truth answers (if in dataset)
data.meta_info       # Metadata dictionary
data.input_ids       # Tokenized prompts
```

## Creating Custom Rewards

### Simple Exact Match

```python
# File: my_reward.py
import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    """Reward for exact match with ground truth."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    responses = data.responses
    ground_truths = data.get('ground_truths', [None] * batch_size)

    for i in range(batch_size):
        if ground_truths[i] is None:
            continue

        # Normalize and compare
        response_ans = responses[i].strip().lower()
        truth_ans = str(ground_truths[i]).strip().lower()

        rewards[i] = 1.0 if response_ans == truth_ans else 0.0

    return rewards
```

### Partial Credit with Similarity

```python
import torch
from difflib import SequenceMatcher
from verl import DataProto

def compute_reward(data: DataProto, use_partial=True, **kwargs) -> torch.Tensor:
    """Reward with optional partial credit."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.get('ground_truths', [None])[i]

        if ground_truth is None:
            continue

        if use_partial:
            # Similarity-based score
            similarity = SequenceMatcher(None, response, ground_truth).ratio()
            rewards[i] = similarity
        else:
            # Exact match only
            rewards[i] = 1.0 if response == ground_truth else 0.0

    return rewards
```

### Math Problem Scoring

```python
import re
import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    """Score math problems by extracting and comparing numeric answers."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.get('ground_truths', [None])[i]

        if ground_truth is None:
            continue

        # Extract numeric answer from response
        response_ans = extract_numeric_answer(response)
        truth_ans = extract_numeric_answer(str(ground_truth))

        if response_ans is not None and truth_ans is not None:
            # Check if close enough (accounting for floating point)
            if abs(response_ans - truth_ans) < 1e-6:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        else:
            rewards[i] = 0.0

    return rewards

def extract_numeric_answer(text: str) -> float:
    """Extract numeric answer from text."""
    # Look for patterns like "answer is 42" or "= 42"
    patterns = [
        r'answer\s*(?:is|:)\s*([-+]?\d+\.?\d*)',
        r'(?:=|≈)\s*([-+]?\d+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass

    # Last resort: find any number
    numbers = re.findall(r'[-+]?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass

    return None
```

### Keyword-Based Scoring

```python
def compute_reward(data: DataProto, keywords=None, **kwargs) -> torch.Tensor:
    """Reward if response contains required keywords."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    keywords = keywords or ['important', 'key', 'critical']

    for i in range(batch_size):
        response = data.responses[i].lower()
        response_lower = response.lower()

        # Count how many keywords are present
        keyword_count = sum(1 for kw in keywords if kw in response_lower)
        proportion = keyword_count / len(keywords)

        rewards[i] = proportion

    return rewards
```

### Code Execution Reward

```python
import subprocess
import tempfile
import torch
from verl import DataProto

def compute_reward(data: DataProto, test_timeout=5, **kwargs) -> torch.Tensor:
    """Score code by testing against test cases."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    test_cases = data.get('test_cases', [None] * batch_size)

    for i in range(batch_size):
        code = data.responses[i]
        tests = test_cases[i]

        if tests is None:
            continue

        # Test code against test cases
        passed_ratio = test_code(code, tests, test_timeout)
        rewards[i] = passed_ratio

    return rewards

def test_code(code: str, test_cases, timeout: int) -> float:
    """Execute code and test against test cases."""
    passed = 0
    total = len(test_cases)

    for test in test_cases:
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name

            # Run with timeout
            result = subprocess.run(
                ['python', code_file],
                input=test['input'],
                capture_output=True,
                timeout=timeout,
                text=True,
            )

            # Check output
            if result.stdout.strip() == test['output'].strip():
                passed += 1

        except subprocess.TimeoutExpired:
            # Timeout = wrong answer
            pass
        except Exception:
            # Runtime error = wrong answer
            pass

    return passed / total if total > 0 else 0.0
```

## Reward Managers

Reward managers control how raw rewards are processed and returned.

### Available Managers

**Naive** (Default)
```bash
reward_model.reward_manager=naive
```
Simple: just compute rewards, no special processing.

**Batch**
```bash
reward_model.reward_manager=batch
```
Process rewards in batches, apply scaling/normalization.

**DAPO**
```bash
reward_model.reward_manager=dapo
```
Divide-and-Conquer Policy Optimization specific handling.

**PRIME**
```bash
reward_model.reward_manager=prime
```
Process Reward Model Ensemble specific handling.

**Limited**
```bash
reward_model.reward_manager=limited
```
Rate-limited reward computation (from experimental.reward.reward_loop).

### Reward Scaling

For stability, normalize rewards:

```yaml
reward_model:
  reward_scaling:
    type: standardize  # or min_max
    mean: 0.0
    std: 1.0
```

Or in custom reward function:

```python
def compute_reward(data: DataProto, scale_rewards=True, **kwargs):
    rewards = ... # raw rewards

    if scale_rewards:
        # Normalize to zero mean, unit variance
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    return rewards
```

## Advantage Estimators

See **references/training.md** → Advantage Estimators for detailed coverage.

Quick reference:

```bash
# GAE (recommended for most tasks)
algorithm.adv_estimator=gae

# GRPO (group-relative, good for ranking)
algorithm.adv_estimator=grpo

# Others
algorithm.adv_estimator=rloo
algorithm.adv_estimator=reinforce_plus_plus
algorithm.adv_estimator=remax
```

## Sandbox-Based Rewards

For code generation tasks, execute generated code in isolated sandbox:

### Setup

Requires running code execution server (e.g., Judge0, custom sandbox):

```bash
reward_model.sandbox_fusion.url=http://localhost:8000
reward_model.sandbox_fusion.max_concurrent=64
reward_model.sandbox_fusion.memory_limit_mb=1024
```

### How It Works

1. Model generates code
2. Code sent to sandbox service
3. Sandbox executes code with test cases
4. Results returned to compute_reward
5. Reward computed from pass/fail

**Location:** `verl/utils/reward_score/sandbox_fusion/`

### Custom Sandbox Integration

```python
import requests
import torch
from verl import DataProto

def compute_reward(
    data: DataProto,
    sandbox_url: str = "http://localhost:8000",
    **kwargs
) -> torch.Tensor:
    """Execute code in sandbox and compute rewards."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        code = data.responses[i]
        test_cases = data.get('test_cases', [None])[i]

        if test_cases is None:
            continue

        # Send to sandbox
        try:
            response = requests.post(
                f"{sandbox_url}/execute",
                json={
                    "code": code,
                    "test_cases": test_cases,
                    "language": "python",
                    "timeout": 5,
                },
                timeout=10,
            )

            result = response.json()

            if result['status'] == 'success':
                # Calculate pass rate
                passed = sum(1 for tc in result['results'] if tc['passed'])
                rewards[i] = passed / len(test_cases)
            else:
                rewards[i] = 0.0

        except Exception as e:
            print(f"Sandbox error: {e}")
            rewards[i] = 0.0

    return rewards
```

## Custom Reward Function Registration

Use `custom_reward_function` config:

```bash
reward_model.custom_reward_function.path=my_reward.py
reward_model.custom_reward_function.name=compute_reward
reward_model.custom_reward_function.reward_kwargs={"scale_rewards": true}
```

The framework:
1. Loads `my_reward.py` dynamically
2. Finds function named `compute_reward`
3. Wraps it with `reward_kwargs`
4. Calls it each training step

**Location:** `verl/trainer/ppo/reward.py:61-115`

## Integration in Training Loop

**File:** `verl/trainer/ppo/ray_trainer.py:1080`

```python
# After rollout generates responses
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)

# Compute rewards
reward_tensor, reward_extra_info = self.reward_fn(gen_batch_output)

# Rewards flow to advantage computation
advantages = compute_advantages(
    reward_tensor=reward_tensor,
    values=critic_output,
    ...
)

# Then to policy loss
policy_loss = compute_policy_loss(
    ...,
    advantages=advantages,
    ...
)
```

The reward_tensor `[batch_size]` becomes input to advantage estimation, which produces advantages `[batch_size, seq_len]` used for policy optimization.

## Debugging Rewards

### Inspect Reward Values

Add to custom reward function:

```python
def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    rewards = torch.zeros(len(data.batch))
    # ... computation ...

    # Debug: log reward statistics
    print(f"Reward stats: mean={rewards.mean()}, std={rewards.std()}, min={rewards.min()}, max={rewards.max()}")

    return rewards
```

### Common Issues

**Rewards all zero:**
- Check ground truth format
- Verify comparison logic
- Test with single example

**Rewards exploding:**
- Add normalization/clipping
- Check reward scaling config

**Model ignores rewards:**
- Ensure rewards have variance
- Check if optimization is happening
- Verify reward is flowing to advantage computation

## Best Practices

1. **Normalize rewards** to roughly [-1, 1] or [0, 1]
2. **Use task-specific scoring** (not random rewards)
3. **Test reward function separately** before training
4. **Monitor reward distribution** during training
5. **Provide dense rewards** when possible (partial credit > binary)
6. **Use curriculum** for complex tasks (easy → hard problems)
