---
name: inf-skill
description: Guide for using the InferenceEngine from final/inf.py. Use when implementing inference, chat, evaluation, or any LLM generation tasks with VERL + SGLang.
---

# InferenceEngine Usage Guide

The `InferenceEngine` class in `final/inf.py` provides a high-level wrapper around VERL's SGLang rollout for easy LLM inference.

## Quick Start

```python
from inf import InferenceEngine

engine = InferenceEngine("config.yaml")
output = engine.generate("What is 2 + 2?")
print(output.completions[0])
```

## Installation

```bash
cd final
bash install.sh
```

## Basic Usage

### Single Prompt (String)

```python
from inf import InferenceEngine

engine = InferenceEngine("config.yaml")

# Simple string prompt
output = engine.generate("Explain quantum computing in one sentence.")
print(output.completions[0])

# Access token IDs
print(f"Generated {len(output.token_ids[0])} tokens")
```

### Batch Prompts (List of Strings)

```python
prompts = [
    "What is the capital of France?",
    "What is 10 * 5?",
    "Name a primary color.",
]

output = engine.generate(prompts)

for prompt, completion in zip(prompts, output.completions):
    print(f"Q: {prompt}")
    print(f"A: {completion}\n")
```

### Multi-Turn Conversation (Message Format)

```python
messages = [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is 5 + 3?"},
    {"role": "assistant", "content": "5 + 3 equals 8."},
    {"role": "user", "content": "Now multiply that by 2."},
]

# Wrap in list for batch of 1
output = engine.generate([messages])
print(output.completions[0])
```

### Batch of Conversations

```python
conversations = [
    [
        {"role": "user", "content": "Hello!"},
    ],
    [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "How are you?"},
    ],
]

output = engine.generate(conversations)
for i, completion in enumerate(output.completions):
    print(f"Conversation {i+1}: {completion}")
```

## Sampling Parameters

### Greedy Decoding (Deterministic)

```python
output = engine.generate(
    "What is the meaning of life?",
    temperature=0.0,
    do_sample=False,
)
```

### Creative Sampling

```python
output = engine.generate(
    "Write a haiku about programming:",
    temperature=0.9,
    top_p=0.95,
    top_k=50,
)
```

### Limit Output Length

```python
output = engine.generate(
    "Count from 1 to 100:",
    max_tokens=50,
)
```

### All Sampling Parameters

```python
output = engine.generate(
    prompt,
    temperature=1.0,    # Sampling temperature (0 = greedy)
    top_p=1.0,          # Nucleus sampling threshold
    top_k=-1,           # Top-k sampling (-1 = disabled)
    max_tokens=512,     # Max new tokens to generate
    do_sample=True,     # Enable/disable sampling
)
```

## Output Structure

```python
output = engine.generate("Hello")

# GenerationOutput dataclass:
output.completions  # List[str] - Generated text
output.token_ids    # List[List[int]] - Token IDs for each completion
output.metadata     # List[Dict] - Additional metadata

# Metadata contains:
# - response_ids: Full response tensor as list
# - response_mask: Which tokens are actual response vs padding
```

## Tool Calling / Function Calling

To enable tool calling, configure `multi_turn` in config.yaml:

```yaml
rollout:
  multi_turn:
    enable: true
    max_assistant_turns: 5
    tool_config_path: "tools.yaml"  # Define your tools here
    format: hermes  # or llama3_json
```

### Tool Config Example (tools.yaml)

```yaml
tools:
  - name: calculator
    description: Perform arithmetic calculations
    parameters:
      type: object
      properties:
        expression:
          type: string
          description: Math expression to evaluate
      required: [expression]
```

### Using with Tools

```python
messages = [
    {"role": "user", "content": "What is 25 * 4?"},
]

output = engine.generate([messages])
# Model may generate tool call in completion
# Parse and execute tool, then continue conversation
```

## Config File Structure

Minimal `config.yaml`:

```yaml
model_path: "Qwen/Qwen3-0.6B"
trust_remote_code: false

rollout:
  name: sglang
  mode: async
  prompt_length: 2048
  response_length: 512
  tensor_model_parallel_size: 1
  gpu_memory_utilization: 0.5
  dtype: auto
```

Full config options in `final/config.yaml`.

## Cleanup

Always shutdown the engine when done:

```python
engine = InferenceEngine("config.yaml")
try:
    output = engine.generate("Hello")
finally:
    engine.shutdown()
```

## Example: GSM8K Evaluation

```python
from datasets import load_dataset
from inf import InferenceEngine

engine = InferenceEngine("config.yaml")
dataset = load_dataset("gsm8k", "main", split="test")

prompts = [f"Solve: {item['question']}" for item in dataset[:10]]
outputs = engine.generate(prompts, temperature=0, do_sample=False)

for prompt, completion in zip(prompts, outputs.completions):
    print(f"{prompt[:50]}... -> {completion[:100]}")

engine.shutdown()
```

## Example: Interactive Chat

```python
from inf import InferenceEngine

engine = InferenceEngine("config.yaml")
history = []

while True:
    user_input = input("You: ")
    if not user_input:
        break

    history.append({"role": "user", "content": user_input})
    output = engine.generate([history])
    assistant_response = output.completions[0]
    history.append({"role": "assistant", "content": assistant_response})
    print(f"AI: {assistant_response}")

engine.shutdown()
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `gpu_memory_utilization` in config (e.g., 0.3)
- Reduce `max_num_seqs` and `max_num_batched_tokens`
- Use smaller batch sizes

### Slow Generation
- Increase `batch_size` for better throughput
- Enable `enable_chunked_prefill: true`
- Enable `enable_prefix_caching: true`

### Empty Completions
- Check `response_length` is sufficient
- Verify model path is correct
- Check tokenizer has proper chat template
