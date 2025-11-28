# Inference: Model Generation & Rollout

How to call model generation using VERL's inference engines and configure sampling parameters.

## Table of Contents

1. Selecting Inference Engine
2. Multi-Turn Conversations
3. Tool/Function Calling
4. Sampling Parameters
5. Custom Rollout Implementations

## Selecting Inference Engine

VERL supports three inference engines via `actor_rollout_ref.rollout.name`:

### SGLang (Recommended)
```bash
actor_rollout_ref.rollout.name=sglang
```
**Best for:**
- Multi-turn conversations
- Tool/function calling
- Complex generation patterns
- Best performance on modern hardware

**Location:** `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

**Key config:**
```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    mode: async  # async or sync
    n: 5  # num_return_sequences
```

### vLLM
```bash
actor_rollout_ref.rollout.name=vllm
```
**Best for:**
- High throughput inference
- Simple generation tasks
- Memory-constrained setups

**Location:** `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

### HuggingFace
```bash
actor_rollout_ref.rollout.name=hf
```
**Best for:**
- Small models
- Development/testing
- Simple setups without external servers

**Location:** `verl/workers/rollout/hf_rollout.py`

## Multi-Turn Conversations

Enable multi-turn conversation support:

### Step 1: Switch to SGLang
```bash
actor_rollout_ref.rollout.name=sglang
```

### Step 2: Set Chat Template
```bash
data.chat_template=chatml  # or alpaca, qwen, etc.
```

### Step 3: Format Data as Conversation
```python
# In your dataset __getitem__
item['messages'] = [
    {"role": "user", "content": "First message"},
    {"role": "assistant", "content": "First response"},
    {"role": "user", "content": "Follow-up question"},
]

# The training loop will use all but the last message as prompt,
# and the last message as the expected response
```

### Step 4: Enable Interactions (Optional)
```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    interactions:
      - conversation
```

### Code Example: Multi-Turn Dataset

```python
from torch.utils.data import Dataset

class MultiTurnDataset(Dataset):
    def __getitem__(self, idx):
        item = self.data[idx]

        # Assume item has 'messages' field
        messages = item['messages']

        # Format conversation: all but last are prompt history
        prompt_messages = messages[:-1]
        response_msg = messages[-1]

        # Format as conversation text
        prompt_text = format_messages(prompt_messages)
        response_text = response_msg['content']

        return {
            'prompt': prompt_text,
            'response': response_text,
            'messages': messages,  # Optional: keep for reference
        }

def format_messages(messages):
    """Format message list into prompt text."""
    text = ""
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        text += f"{role}: {content}\n"
    return text
```

## Tool/Function Calling

Enable model to call external tools/functions:

### Step 1: Define Tools
```yaml
# tools.yaml
tools:
  - name: "calculator"
    description: "Add two numbers together"
    parameters:
      - name: "a"
        type: "number"
        description: "First number"
      - name: "b"
        type: "number"
        description: "Second number"

  - name: "search"
    description: "Search the internet"
    parameters:
      - name: "query"
        type: "string"
        description: "Search query"
```

### Step 2: Configure Tools in Training
```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.tools=tools.yaml
```

### Step 3: SGLang Handles Tool Parsing
SGLang automatically parses generated tool calls and returns them in structured format. The framework integrates tool parsing via `FunctionCallParser`.

**Location:** `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~2000`

### Code Example: Using Tool Calls in Reward
```python
def compute_reward(data: DataProto, **kwargs):
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]

        # Check if model called the right tool
        if has_correct_tool_call(response):
            rewards[i] = 1.0
        else:
            rewards[i] = 0.0

    return rewards

def has_correct_tool_call(response: str):
    """Check if response contains correct tool invocation."""
    import re
    # Pattern: <TOOL>calculator</TOOL>...parameters
    return bool(re.search(r'<TOOL>calculator</TOOL>', response))
```

## Sampling Parameters

Configure how the model generates responses:

### Key Parameters

```bash
# Temperature: controls randomness
# 0 = deterministic, 1 = normal, >1 = very random
actor_rollout_ref.rollout.temperature=0.7

# Top-P (nucleus sampling): only sample from top P% of probability mass
actor_rollout_ref.rollout.top_p=0.9

# Top-K: only sample from top K tokens
actor_rollout_ref.rollout.top_k=50

# Max length for response
actor_rollout_ref.rollout.max_response_length=1024

# Number of return sequences (per prompt)
actor_rollout_ref.rollout.n=5
```

### Recommended Settings

**Deterministic (for testing):**
```bash
temperature=0.1
top_p=0.95
top_k=50
```

**Balanced (standard):**
```bash
temperature=0.7
top_p=0.9
top_k=50
```

**Diverse (exploration):**
```bash
temperature=1.0
top_p=0.9
top_k=50
```

## Custom Rollout Implementations

Create custom inference logic by extending `BaseRollout`:

```python
# File: my_rollout.py
from verl.workers.rollout.base import BaseRollout
from verl import DataProto
import torch

class CustomRollout(BaseRollout):
    """Custom inference implementation."""

    def __init__(self, config, role):
        super().__init__(config, role)
        self.config = config
        # Initialize your inference engine here
        self.engine = init_custom_engine(config)

    def generate_sequences(self, batch: DataProto) -> DataProto:
        """
        Generate sequences for input batch.

        Args:
            batch: DataProto with prompts and metadata

        Returns:
            DataProto with responses and log probabilities added
        """
        # Extract prompts
        prompts = batch.prompts
        batch_size = len(batch.batch)

        # Call inference engine
        outputs = self.engine.generate(
            prompts,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_length=self.config.max_response_length,
            num_return_sequences=self.config.n,
        )

        # Extract responses and log probs
        responses = outputs['responses']  # List of strings
        log_probs = outputs['log_probs']  # Tensor [batch, seq_len]

        # Add to batch
        output_batch = batch.clone()
        output_batch.responses = responses
        output_batch.log_probs = log_probs
        output_batch.response_lengths = torch.tensor([len(r) for r in responses])

        return output_batch

    def compute_log_probs(self, ...):
        """Compute log probabilities for reference policy (if needed)."""
        pass
```

### Register Custom Rollout
```bash
actor_rollout_ref.rollout.name=custom \
actor_rollout_ref.rollout.custom_cls_path=my_rollout.py \
actor_rollout_ref.rollout.custom_cls_name=CustomRollout
```

## Entry Point in Training Loop

The inference is called during training at:

**File:** `verl/trainer/ppo/ray_trainer.py:1055`

```python
# Synchronous generation
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)

# Or asynchronous generation
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

The generated `gen_batch_output` is a `DataProto` containing:
- `responses`: Generated text
- `log_probs`: Log probability of each token
- `response_lengths`: Length of each response
- Original batch data (prompts, etc.)
