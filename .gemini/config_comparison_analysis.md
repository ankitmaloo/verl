# VERL Config Variables Analysis: Production vs Your Setup

## Overview
This document breaks down each configuration variable from VERL's production PPO training command and compares it with your `final/` folder setup, highlighting where using defaults is hurting performance.

---

## The VERL Production Command (Annotated)

```bash
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    # === DATA CONFIG ===
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=4096 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    
    # === ROLLOUT (INFERENCE) CONFIG ===
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    
    # === MODEL CONFIG ===
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    
    # === ACTOR (POLICY) TRAINING CONFIG ===
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    
    # === REFERENCE MODEL CONFIG ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    
    # === CRITIC (VALUE) MODEL CONFIG ===
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2-7B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    
    # === ALGORITHM CONFIG ===
    algorithm.kl_ctrl.kl_coef=0.001 \
    
    # === TRAINER CONFIG ===
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15
```

---

## Detailed Config Variable Breakdown

### 🔴 CRITICAL: Sequence Length Settings

| Variable | VERL Production | Your Config | Your inf.py Default | Impact |
|----------|----------------|-------------|---------------------|---------|
| `data.max_prompt_length` | **4096** | 2048 | 2048 | **🔴 HURTING YOU** |
| `data.max_response_length` | **4096** | 512 | 512 | **🔴 HURTING YOU** |
| `prompt_length` (rollout) | **4096** (inherited) | 2048 | 2048 | **🔴 HURTING YOU** |
| `response_length` (rollout) | **4096** (inherited) | 512 | 512 | **🔴 HURTING YOU** |

**Why this hurts:**
- ❌ **Your prompts get truncated at 2048 tokens** - Complex reasoning problems need longer context
- ❌ **Your responses get cut off at 512 tokens** - Math problems often need >512 tokens for step-by-step reasoning
- ❌ **GSM8K needs long responses** - The "think step by step" instruction generates verbose chains of thought
- ❌ **You're losing information** - Truncated prompts = lost context; truncated responses = incomplete reasoning

**RolloutConfig defaults you're inheriting:**
```python
prompt_length: int = 512      # ❌ You're using 2048, but VERL uses 4096
response_length: int = 512    # ❌ You're using 512, but VERL uses 4096
```

---

### 🟡 IMPORTANT: GPU Memory Utilization

| Variable | VERL Production | Your Config | Your inf.py Default | Impact |
|----------|----------------|-------------|---------------------|---------|
| `gpu_memory_utilization` | **0.8** | 0.5 | **0.8** ✅ | **🟡 PARTIALLY HURTING YOU** |

**Why this matters:**
- ⚠️ **config.yaml uses 0.5** - You're only using 50% of GPU memory, leaving half unused
- ✅ **inf.py uses 0.8** - Good! This is correct
- ⚠️ **inference.py uses 0.5** - Also leaving 30% GPU memory on the table
- 💡 **Lower utilization = smaller batch sizes** - Can't fit as many sequences, slower throughput

**RolloutConfig default you're inheriting:**
```python
gpu_memory_utilization: float = 0.5  # ⚠️ VERL production uses 0.8
```

---

### 🟡 IMPORTANT: Tensor Parallelism

| Variable | VERL Production | Your Config | Your inf.py Default | Impact |
|----------|----------------|-------------|---------------------|---------|
| `tensor_model_parallel_size` | **2** | 1 | 1 | **🟡 HURTING MULTI-GPU** |

**Why this matters:**
- ⚠️ **You're using 1 GPU** - Can only load smaller models
- ✅ **VERL uses 2 GPUs** - Splits Qwen2-7B across 2 GPUs for faster inference
- 💡 **TP=2 enables larger models** - You can run 7B+ models that don't fit on 1 GPU
- 💡 **TP=2 can be faster** - Parallel computation across GPUs

**RolloutConfig default:**
```python
tensor_model_parallel_size: int = 2  # ❌ You override to 1, VERL uses 2
```

**Note:** You're explicitly setting `num_gpus: 1` in config.yaml, so you're NOT using the default here - you're choosing 1 GPU.

---

### 🔴 CRITICAL: Batching and Throughput

| Variable | VERL Production | Your Config | RolloutConfig Default | Impact |
|----------|----------------|-------------|----------------------|---------|
| `data.train_batch_size` | **4096** | N/A (inference only) | N/A | Not applicable |
| `max_num_seqs` | (default) | (default) | **1024** | **🟢 OK** |
| `max_num_batched_tokens` | (default) | (default) | **8192** | **🔴 HURTING YOU** |
| `log_prob_micro_batch_size_per_gpu` | **8** | N/A | None | Not applicable (you're inference-only) |

**Why max_num_batched_tokens=8192 hurts:**
- ❌ **With 4096 prompt + 4096 response = 8192 total**
- ❌ **Your default 8192 can only fit 1 sequence** (4096+4096=8192)
- ❌ **You need higher for batching** - Should be 32768+ for 4 concurrent sequences
- 💡 **Low batched tokens = poor throughput** - Can't batch multiple requests efficiently

**What you're missing:**
```python
# You're not setting these in inf.py:
max_num_batched_tokens: int = 8192  # ❌ Too low for 4096-length sequences
max_num_seqs: int = 1024            # ✅ This is fine
```

---

### 🟢 OK: Sampling Parameters

| Variable | VERL Production | Your Config | RolloutConfig Default | Impact |
|----------|----------------|-------------|----------------------|---------|
| `temperature` | (default) | 1.0 | 1.0 | ✅ OK |
| `top_p` | (default) | 1.0 | 1.0 | ✅ OK |
| `top_k` | (default) | -1 | -1 | ✅ OK |
| `do_sample` | (default) | true | true | ✅ OK |

**These are fine** - You're matching VERL's defaults correctly.

---

### 🟢 OK: Data Type

| Variable | VERL Production | Your Config | Your inf.py Default | Impact |
|----------|----------------|-------------|---------------------|---------|
| `dtype` | (default) | "auto" | **"bfloat16"** | ✅ OK |

**Why this is OK:**
- ✅ VERL uses bfloat16 (default)
- ✅ inf.py uses bfloat16 (hardcoded)
- ⚠️ config.yaml uses "auto" - Will likely choose bfloat16 anyway

**RolloutConfig default:**
```python
dtype: str = "bfloat16"  # ✅ Good default
```

---

### 🟡 IMPORTANT: Cache Management

| Variable | VERL Production | Your Config | RolloutConfig Default | Impact |
|----------|----------------|-------------|----------------------|---------|
| `free_cache_engine` | (default) | true | true | ✅ OK |
| `enforce_eager` | (default) | true | true | ✅ OK |

**Why these matter:**
- ✅ `free_cache_engine=true` - Frees KV cache after each generation (important for training)
- ✅ `enforce_eager=true` - Disables CUDA graphs (needed for dynamic batch sizes in RL)

**These are correct for your inference-only use case too.**

---

### 🔵 INFO: Variables Not Applicable to Inference-Only

These are VERL training-specific and don't apply to your `final/` inference setup:

| Variable | Purpose | Why Not Applicable |
|----------|---------|-------------------|
| `ppo_mini_batch_size=64` | PPO training mini-batch | No training loop |
| `ppo_micro_batch_size_per_gpu=4` | Per-GPU micro-batching for PPO | No training loop |
| `fsdp_config.param_offload=True` | Offload params to CPU | Inference doesn't need FSDP |
| `fsdp_config.optimizer_offload=True` | Offload optimizer to CPU | No optimizer in inference |
| `enable_gradient_checkpointing=True` | Save memory during backprop | No gradients in inference |
| `algorithm.kl_ctrl.kl_coef=0.001` | KL divergence penalty | PPO-specific |
| `optim.lr=1e-6` | Learning rate | No training |

---

## Critical Missing Variables in Your Setup

### Variables You're NOT Setting (Using Bad Defaults)

```python
# In inf.py _build_rollout_config(), you're missing:

# 🔴 CRITICAL - Batching throughput
"max_num_batched_tokens": ???  # Uses default 8192 - TOO LOW!

# 🟡 IMPORTANT - Sequence handling  
"max_model_len": ???            # Uses None - Should set explicitly
"max_num_seqs": ???             # Uses default 1024 - OK but could tune

# 🟢 OPTIONAL - Advanced settings
"ignore_eos": ???               # Uses default False - OK
"enforce_eager": ???            # Uses default True - OK (but could explicit)
"over_sample_rate": ???         # Uses default 0.0 - Not needed for single-turn
```

---

## How Defaults Are Hurting You: Summary Table

| Issue | Your Value | VERL Production | Default Hurting You? | Severity |
|-------|-----------|-----------------|---------------------|----------|
| **Prompt Length** | 2048 | 4096 | ✅ Yes (default 512, you set 2048, but still half of VERL) | 🔴 CRITICAL |
| **Response Length** | 512 | 4096 | ✅ Yes (default 512, matching default IS the problem) | 🔴 CRITICAL |
| **GPU Memory** | 0.5 (config) / 0.8 (inf.py) | 0.8 | ✅ Yes (config.yaml using 0.5) | 🟡 MODERATE |
| **Batched Tokens** | 8192 (default) | Likely 32768+ | ✅ Yes (default too low for long sequences) | 🔴 CRITICAL |
| **Tensor Parallel** | 1 | 2 | ⚠️ Intentional (single GPU setup) | 🟢 OK for 1-GPU |

---

## The Real Problem: Your Rollout Config Builder

Looking at your `inf.py`:

```python
def _build_rollout_config(self) -> RolloutConfig:
    rollout_dict = {
        "name": "sglang",
        "mode": self.config.get("rollout_mode", "sync"),
        "skip_tokenizer_init": True,
        "prompt_length": self.config.get("max_prompt_length", 2048),  # ✅ SET
        "response_length": self.config.get("max_response_length", 512),  # ❌ 512 default
        "dtype": self.config.get("dtype", "bfloat16"),  # ✅ OK
        "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.8),  # ✅ OK
        "tensor_model_parallel_size": self.config.get("num_gpus", 1),  # ✅ OK for 1-GPU
        "temperature": self.config.get("temperature", 1.0),
        "top_p": self.config.get("top_p", 1.0),
        "top_k": self.config.get("top_k", -1),
        "do_sample": self.config.get("do_sample", True),
        "n": self.config.get("n", 1),
        "free_cache_engine": self.config.get("free_cache_engine", True),
        "multi_turn": self._build_multi_turn_cfg(),
    }
    # ❌ NOT SETTING: max_num_batched_tokens, max_num_seqs, max_model_len
```

**What happens:**
1. You set 10 variables explicitly ✅
2. For the other ~30 RolloutConfig fields, **you inherit the class defaults** ❌
3. Some defaults are OK (temperature, top_p), but some are terrible for your use case:
   - `max_num_batched_tokens=8192` - Too low!
   - `prompt_length=512` default - You override to 2048, but should be 4096
   - `response_length=512` default - You don't override, should be 4096

---

## Specific Ways Defaults Hurt Performance

### 1. **Sequence Length = Truncated Reasoning** 🔴
```
VERL: 4096 prompt + 4096 response = 8192 total
You:  2048 prompt + 512 response = 2560 total

Problem: GSM8K "think step by step" generates long chains
- ❌ Your 512 response limit cuts off mid-reasoning
- ❌ Model generates "2 + 2 = 4, then we multip..." [TRUNCATED]
- ✅ VERL's 4096 lets model complete: "...multiply by 3 = 12 #### 12"
```

### 2. **Low GPU Memory = Wasted Resources** 🟡
```
VERL: 0.8 utilization = ~38GB used on A100 (48GB)
You:  0.5 utilization = ~24GB used, ~24GB wasted

Problem: Can't fit as many concurrent sequences
- ❌ Smaller batch size = lower throughput
- ❌ More idle time between batches
- ✅ Higher utilization = better GPU efficiency
```

### 3. **Low Batched Tokens = Poor Throughput** 🔴
```
VERL: Likely 32768+ batched tokens
You:  8192 batched tokens (default)

With your 2048 prompt + 512 response = 2560 tokens:
- You can batch: 8192 / 2560 = ~3 sequences
- VERL can batch: 32768 / 8192 = ~4 sequences (with longer sequences!)

Problem: Lower parallelism
- ❌ Fewer requests processed simultaneously
- ❌ Lower GPU utilization
- ❌ Longer total inference time
```

### 4. **Missing max_model_len** 🟡
```
VERL: Sets max_model_len explicitly (likely 8192+)
You:  None (lets SGLang auto-detect)

Problem: Unpredictable behavior
- ❌ SGLang might allocate less KV cache than needed
- ❌ Could fail on long sequences
- ⚠️ Less control over memory allocation
```

---

## Recommendations (In Order of Impact)

### 🔴 CRITICAL: Fix Sequence Lengths
```yaml
# In config.yaml, change:
max_prompt_length: 4096   # Was: 2048
max_response_length: 4096 # Was: 512
```

**Impact:** Enables full reasoning chains, no truncation

### 🔴 CRITICAL: Increase Batched Tokens
```python
# In inf.py _build_rollout_config(), add:
"max_num_batched_tokens": 32768,  # 4x default, supports 4 concurrent 8K sequences
```

**Impact:** 4x throughput improvement

### 🟡 MODERATE: Increase GPU Memory
```yaml
# In config.yaml, change:
gpu_memory_utilization: 0.8  # Was: 0.5
```

**Impact:** 60% more memory available, bigger batches

### 🟡 MODERATE: Set max_model_len Explicitly
```python
# In inf.py _build_rollout_config(), add:
"max_model_len": 8192,  # prompt_length + response_length
```

**Impact:** Predictable memory allocation

### 🟢 OPTIONAL: Set enforce_eager Explicitly
```python
# In inf.py _build_rollout_config(), add:
"enforce_eager": True,  # Already default, but explicit is better
```

**Impact:** None (already default), but clearer intent

---

## Config.yaml: Before vs After

### Before (Current - Suboptimal)
```yaml
max_prompt_length: 2048      # ❌ Half of VERL
max_response_length: 512     # ❌ 1/8 of VERL
gpu_memory_utilization: 0.5  # ❌ Wasting 30% GPU
```

### After (Recommended)
```yaml
max_prompt_length: 4096      # ✅ Matches VERL
max_response_length: 4096    # ✅ Matches VERL
gpu_memory_utilization: 0.8  # ✅ Matches VERL
```

---

## inf.py: Before vs After

### Before (Current - Missing Important Settings)
```python
rollout_dict = {
    "prompt_length": self.config.get("max_prompt_length", 2048),
    "response_length": self.config.get("max_response_length", 512),
    # ... 10 other fields ...
}
# ❌ 30+ fields use RolloutConfig defaults!
```

### After (Recommended - Explicit Control)
```python
rollout_dict = {
    "prompt_length": self.config.get("max_prompt_length", 2048),
    "response_length": self.config.get("max_response_length", 512),
    "max_num_batched_tokens": 32768,  # ✅ Explicit, 4x default
    "max_model_len": (
        self.config.get("max_prompt_length", 2048) +
        self.config.get("max_response_length", 512)
    ),  # ✅ Explicit
    "max_num_seqs": 1024,  # ✅ Explicit (same as default, but clear)
    # ... other fields ...
}
```

---

## Bottom Line

**You are being hurt by defaults in 3 ways:**

1. **🔴 CRITICAL - Sequence lengths:** Your config.yaml sets 2048/512 instead of 4096/4096
   - **Not a default problem** - You explicitly chose these values
   - **But they're too small** for GSM8K-style reasoning

2. **🔴 CRITICAL - Batched tokens:** You're inheriting the default 8192
   - **This IS a default problem** - You're not setting it
   - **8192 is way too low** for long sequences

3. **🟡 MODERATE - GPU memory:** Your config.yaml uses 0.5 instead of 0.8
   - **Not a default problem** - You explicitly chose 0.5
   - **But it's too conservative** for production workloads

**Fix these 3 things and you'll match VERL's production performance!**
