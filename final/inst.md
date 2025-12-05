# VERL SGLang Inference Engine - Instructions for LLM Agents

> **Purpose**: This document provides complete context for understanding and working with the `inf.py` inference engine that uses VERL's SGLangRollout in async mode.

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Key Files & Their Roles](#key-files--their-roles)
4. [Data Flow: Token-In-Token-Out](#data-flow-token-in-token-out)
5. [Configuration Reference](#configuration-reference)
6. [How inf.py Works](#how-infpy-works)
7. [SGLang Async Mode Explained](#sglang-async-mode-explained)
8. [Single GPU Execution Flow](#single-gpu-execution-flow)
9. [Tool Calling Support](#tool-calling-support)
10. [Common Issues & Solutions](#common-issues--solutions)
11. [Quick Reference: Code Patterns](#quick-reference-code-patterns)

---

## Executive Summary

**What is this?**
- `inf.py` is a lightweight inference wrapper that uses VERL's `SGLangRollout` class for efficient LLM inference
- It operates in **async mode** using SGLang's `AsyncEngine` for concurrent request processing
- Designed for **single GPU** setups but architecture supports multi-GPU

**Key Design Decisions:**
1. Uses VERL's `SGLangRollout` (not raw `sgl.Engine`) for RL training integration
2. Token-in-token-out mode (`skip_tokenizer_init: true`) - tokenization happens in wrapper, not engine
3. Async mode processes requests concurrently via `asyncio.gather()`
4. Configuration lives in `config.yaml` under the `rollout:` section

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Application                             │
│                                                                       │
│    prompts = ["What is 2+2?", "Explain AI"]   # Text input          │
│    outputs = engine.generate(prompts)          # Text output         │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     inf.py :: InferenceEngine                        │
│                                                                       │
│  1. _normalize_prompts() → Converts to List[List[Message]]          │
│  2. _build_dataproto()   → Tokenizes to tensor of token IDs         │
│  3. rollout.generate_sequences() → Calls SGLangRollout              │
│  4. _decode_output()     → Converts token IDs back to text          │
└────────────────────────────┬───────────────────────────────────────┘
                             │ DataProto with token IDs
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              VERL :: SGLangRollout (sglang_rollout.py)              │
│                                                                       │
│  • Manages distributed/single GPU execution                          │
│  • Handles async request processing with asyncio.gather()           │
│  • Interfaces with SGLang's AsyncEngine                              │
│  • Supports tool calling & multi-turn conversations                  │
└────────────────────────────┬───────────────────────────────────────┘
                             │ List[token_ids]
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SGLang :: AsyncEngine                           │
│                                                                       │
│  • CUDA kernel execution                                             │
│  • KV cache management                                               │
│  • FlashAttention (fa3)                                              │
│  • Returns generated token IDs                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Files & Their Roles

| File | Purpose | Status |
|------|---------|--------|
| `final/inf.py` | Main inference wrapper - **USE THIS** | ✅ Production-ready |
| `final/config.yaml` | Configuration for rollout settings | ✅ Complete |
| `final/inference.py` | Old implementation using raw `sgl.Engine` | ⚠️ **DEPRECATED** - bypasses VERL |
| `final/simple_inference.py` | Minimal SGLang example | ⚠️ For testing only |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | VERL's core rollout class | ✅ Source of truth |
| `verl/workers/config/rollout.py` | RolloutConfig dataclass definition | Reference |

### Why `inf.py` over `inference.py`?

| Feature | inf.py | inference.py |
|---------|--------|--------------|
| Uses VERL's SGLangRollout | ✅ Yes | ❌ No (raw sgl.Engine) |
| RL training integration | ✅ Ready | ❌ Not possible |
| Weight sync support | ✅ Yes | ❌ No |
| Multi-turn/Tool calling | ✅ Via config | ⚠️ Manual implementation |
| Async mode | ✅ Default | ❌ Not used |
| Config structure | ✅ Nested `rollout:` | ⚠️ Flat keys |

---

## Data Flow: Token-In-Token-Out

This is **critical** to understand. The entire pipeline works with token IDs, not text:

```
                    YOUR CODE
                        │
                        ▼
    ┌───────────────────────────────────────────┐
    │  Input: "What is 2+2?"                    │
    │  ↓ tokenizer.encode()                     │
    │  Token IDs: [3555, 374, 220, 17, 10, 17, 30] │
    └───────────────────┬───────────────────────┘
                        │
                        ▼ DataProto.batch["input_ids"]
    ┌───────────────────────────────────────────┐
    │           SGLangRollout                   │
    │                                           │
    │  • Strips left padding                    │
    │  • Sends to SGLang engine as token IDs   │
    │  • NO TEXT at engine level!              │
    └───────────────────┬───────────────────────┘
                        │
                        ▼ input_ids=[3555, 374, ...]
    ┌───────────────────────────────────────────┐
    │           SGLang AsyncEngine              │
    │                                           │
    │  • Generates: [220, 19] (= " 4")         │
    │  • Returns token IDs                      │
    └───────────────────┬───────────────────────┘
                        │
                        ▼ DataProto.batch["responses"]
    ┌───────────────────────────────────────────┐
    │           inf.py :: _decode_output()      │
    │                                           │
    │  • responses: torch.Tensor([220, 19])     │
    │  • .tolist() → [220, 19]                  │
    │  • tokenizer.decode() → " 4"              │
    └───────────────────┬───────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────┐
    │  Output: GenerationOutput                 │
    │    - completions: [" 4"]                  │
    │    - token_ids: [[220, 19]]               │
    └───────────────────────────────────────────┘
```

**Key Setting:** `skip_tokenizer_init: true` enables this token-in-token-out mode.

---

## Configuration Reference

### Minimal Working Config (Single GPU)

```yaml
model_path: "Qwen/Qwen3-0.6B"

rollout:
  # REQUIRED for async mode
  name: sglang
  mode: async
  skip_tokenizer_init: true
  
  # Sequence lengths
  prompt_length: 2048
  response_length: 512
  
  # Single GPU
  tensor_model_parallel_size: 1
  
  # Performance
  gpu_memory_utilization: 0.5
  dtype: auto
  enforce_eager: false  # Use CUDA graphs
```

### Full Config Structure

```yaml
rollout:
  # ─── Core Settings (REQUIRED) ───
  name: sglang
  mode: async                    # "async" or "sync" (deprecated)
  skip_tokenizer_init: true      # REQUIRED for async
  
  # ─── Sequence Lengths ───
  prompt_length: 2048
  response_length: 512
  max_model_len: null            # Auto: prompt_length + response_length
  max_num_seqs: 1024
  max_num_batched_tokens: 8192
  
  # ─── Sampling (Training) ───
  temperature: 1.0
  top_k: -1                      # -1 = disabled
  top_p: 1.0
  do_sample: true
  n: 1
  ignore_eos: false
  
  # ─── Sampling (Validation) ───
  val_kwargs:
    temperature: 0
    do_sample: false
  
  # ─── Parallelism ───
  tensor_model_parallel_size: 1  # TP
  data_parallel_size: 1          # DP
  expert_parallel_size: 1        # EP (MoE)
  pipeline_model_parallel_size: 1 # PP
  
  # ─── Memory & Performance ───
  gpu_memory_utilization: 0.5
  dtype: auto                    # bfloat16/float16/float32/auto
  free_cache_engine: true
  enforce_eager: false           # false = CUDA graphs (faster)
  enable_chunked_prefill: true
  enable_prefix_caching: true
  
  # ─── Engine Settings ───
  load_format: dummy             # "dummy" or "auto"
  engine_kwargs:
    sglang:
      attention_backend: fa3
  
  # ─── Multi-turn / Tools ───
  multi_turn:
    enable: false
    tool_config_path: null
    format: hermes
    max_assistant_turns: null
  
  # ─── Advanced ───
  over_sample_rate: 0.0
  calculate_log_probs: false
  disable_log_stats: true
```

---

## How inf.py Works

### Initialization Flow

```python
class InferenceEngine:
    def __init__(self, config_path: str):
        # 1. Load YAML config
        self.config = yaml.safe_load(open(config_path))
        
        # 2. Initialize distributed (single GPU defaults)
        _ensure_dist_initialized()  # Sets RANK=0, WORLD_SIZE=1
        
        # 3. Load HuggingFace model config & tokenizer
        self.model_config = HFModelConfig(path=config["model_path"])
        self.tokenizer = self.model_config.tokenizer
        
        # 4. Build RolloutConfig from nested config
        rollout_conf = self._build_rollout_config()
        
        # 5. Create SGLangRollout (this starts the engine!)
        self.rollout = SGLangRollout(
            config=rollout_conf,
            model_config=self.model_config,
            device_mesh=None,  # Single GPU, no mesh needed
        )
```

### Generation Flow

```python
def generate(self, prompts, **kwargs):
    # 1. Normalize input to List[List[Message]]
    message_batches = self._normalize_prompts(prompts)
    # Result: [[{"role": "user", "content": "What is 2+2?"}], ...]
    
    # 2. Build DataProto with token IDs
    data_proto = self._build_dataproto(message_batches)
    # Result: DataProto with batch["input_ids"] tensor
    
    # 3. Apply sampling params
    sampling_params, do_sample = self._sampling_kwargs(**kwargs)
    
    # 4. Call SGLangRollout (the magic happens here!)
    output = self.rollout.generate_sequences(data_proto, **sampling_params)
    
    # 5. Decode token IDs to text
    return self._decode_output(output)
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `_build_rollout_config()` | Converts config.yaml → RolloutConfig dataclass |
| `_build_dataproto()` | Tokenizes text → DataProto with tensors |
| `_normalize_prompts()` | Converts various input formats to messages |
| `_decode_output()` | Converts response token IDs → text |
| `_sampling_kwargs()` | Extracts sampling params from kwargs |

---

## SGLang Async Mode Explained

### Sync vs Async Mode

| Aspect | Sync Mode (Deprecated) | Async Mode (Recommended) |
|--------|------------------------|--------------------------|
| Engine | `sglang.LLM` | `sglang.AsyncEngine` |
| Processing | Batch as single unit | Concurrent requests |
| Tool calling | ❌ Limited | ✅ Full support |
| Multi-turn | ❌ Limited | ✅ Full support |
| Early stopping | ❌ No | ✅ Via `over_sample_rate` |

### How Async Processes Batches

```python
# When you send 100 prompts:

# 1. Convert to 100 AsyncRolloutRequest objects
req_list = [AsyncRolloutRequest(prompt_ids=ids) for ids in all_prompts]

# 2. Launch ALL 100 concurrently with asyncio.gather()
output_req_list = await asyncio.gather(
    *[self._async_rollout_a_request(req) for req in req_list]
)

# 3. SGLang engine batches internally for GPU efficiency
# 4. Results return as they complete
# 5. Sort and return all 100 outputs
```

**Key Insight:** You still give batches! Async mode just processes them as concurrent individual requests, giving flexibility for tool calling and early stopping.

---

## Single GPU Execution Flow

With `tensor_model_parallel_size: 1`, here's what happens:

```python
# Initialization
self._rank = 0       # Only rank
self._tp_rank = 0    # Only TP rank
self._tp_size = 1    # Single GPU

# Engine creation (only rank 0 creates engine)
if effective_first:  # TRUE for single GPU
    self._engine = AsyncEngine(
        model_path=actor_module,
        tp_size=1,
        node_rank=0,
        nnodes=1,
    )

# Generation
if self._tp_rank == 0:  # TRUE - we run inference
    req_list = self._preprocess_prompts(prompts)
    outputs = await asyncio.gather(*[...])  # Process all
    
dist.barrier()  # No-op for single GPU
```

**All distributed code becomes no-ops with 1 GPU** - barriers pass through, broadcasts return same data.

---

## Tool Calling Support

### Enabling Tools

```yaml
rollout:
  multi_turn:
    enable: true                    # Turn on!
    tool_config_path: tools.yaml    # Your tool definitions
    format: hermes                  # Tool call format
    max_assistant_turns: 5          # Max tool calls
```

### Tool Config File Example

```yaml
# tools.yaml
- class_path: verl.tools.gsm8k_tool.Gsm8kTool
  class_name: Gsm8kTool
  tool_schema:
    type: function
    function:
      name: calculate_answer
      description: "Submit your final numerical answer"
      parameters:
        type: object
        properties:
          answer:
            type: string
            description: "The numerical answer"
        required:
          - answer
```

### Built-in Tools

| Tool | Purpose |
|------|---------|
| `Gsm8kTool` | Math problem solving with reward |
| `Geo3kTool` | Geometry problems |
| `SearchTool` | Web/document search |
| `ImageZoomInTool` | Image region zoom |
| `SandboxFusionTools` | Code execution |

### Tool Execution Flow

```python
# Inside SGLangRollout:
if self._function_call_parser.has_tool_call(content):
    # Parse tool calls from output
    tool_calls = self._function_call_parser.parse_non_stream(content)
    
    # Execute ALL tools concurrently
    results = await asyncio.gather(*[
        self._tool_map[call.function.name].execute(...)
        for call in tool_calls
    ])
    
    # Add tool responses to conversation
    # Continue generation
```

---

## Common Issues & Solutions

### 1. "async mode requires skip_tokenizer_init to be True"

```yaml
# Fix:
rollout:
  mode: async
  skip_tokenizer_init: true  # MUST be true for async
```

### 2. Config not being read correctly

```python
# Old flat config (deprecated):
self.config.get("rollout_mode", "sync")  # ❌

# New nested config:
self.config.get("rollout", {}).get("mode", "async")  # ✅
```

### 3. CUDA out of memory

```yaml
rollout:
  gpu_memory_utilization: 0.3  # Reduce from 0.5
  enforce_eager: true          # Disable CUDA graphs
  max_num_seqs: 256            # Reduce batch capacity
```

### 4. SGLang >= 0.5.5 deprecation warning

The old SPMD mode is deprecated. Always use:
```yaml
rollout:
  mode: async  # NOT "sync"
```

### 5. Missing libnvrtc.so.12

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If issues, set library path:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Quick Reference: Code Patterns

### Basic Usage

```python
from final.inf import InferenceEngine

engine = InferenceEngine("final/config.yaml")

# Single prompt
output = engine.generate("What is 2+2?")
print(output.completions[0])

# Batch of prompts
outputs = engine.generate([
    "What is 2+2?",
    "Explain quantum computing",
    "Write a haiku about coding"
])
for completion in outputs.completions:
    print(completion)
```

### With Custom Sampling

```python
output = engine.generate(
    "Creative story about AI",
    temperature=0.8,
    top_p=0.95,
    max_tokens=1000
)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
    {"role": "user", "content": "What's the weather like?"}
]
output = engine.generate([messages])
```

### Accessing Token IDs

```python
output = engine.generate("Hello world")

# Text output
print(output.completions[0])

# Token IDs
print(output.token_ids[0])  # [220, 19, ...]

# Full metadata
print(output.metadata[0])   # {"response_ids": [...], "response_mask": [...]}
```

---

## Summary: What You Need to Know

1. **Use `inf.py`** - It's the production-ready wrapper using VERL's SGLangRollout
2. **Config lives in `rollout:` section** - Nested structure, not flat keys
3. **Async mode is default** - Concurrent request processing via `asyncio.gather()`
4. **Token-in-token-out** - Tokenization in wrapper, not engine (`skip_tokenizer_init: true`)
5. **Single GPU works fine** - All distributed code gracefully degrades
6. **Tool calling is built-in** - Just enable `multi_turn.enable` and provide tools
7. **Integrates with RL training** - Weight sync ready, not bypassed like `inference.py`

---

*Document generated from chat conversation about VERL SGLang inference setup.*
*Last updated: 2025-12-05*
