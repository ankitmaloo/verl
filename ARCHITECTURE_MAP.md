# VERL RL Pipeline Architecture Map

## Quick Summary: How `run_qwen3-8b.sh` Works

```
run_qwen3-8b.sh
  └─> python3 -m verl.trainer.main_ppo [config overrides]
      └─> main_ppo.py:main() [Hydra entry point]
          └─> main_ppo.py:run_ppo()
              └─> Ray cluster initialization
              └─> TaskRunner.run()
                  ├─> Dataset loading (ENV)
                  ├─> Reward manager loading (ALGO: reward_fn)
                  ├─> Worker initialization (TRAINING: actor/critic, INFERENCE: rollout)
                  └─> RayPPOTrainer.fit()
                      └─> Training loop (epoch/batch)
                          ├─> INFERENCE: generate_sequences() → SGLang/vLLM
                          ├─> ALGO: compute_reward() → reward function
                          ├─> ALGO: compute advantages (GAE/GRPO)
                          ├─> TRAINING: backward pass → actor/critic
                          └─> Logging & checkpointing
```

---

## The Four Components

### 1. **INFERENCE** (Where generation/rollout happens)

**Primary Entry Point:**
- `verl/trainer/ppo/ray_trainer.py:1055` → `self.actor_rollout_wg.generate_sequences(gen_batch_output)`

**Source Code Locations:**

| Inference Type | Source File | Key Class | Method |
|---|---|---|---|
| **SGLang** (recommended for multi-turn) | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | `SglangRollout` | `generate_sequences()` |
| **vLLM** | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | `VllmRollout` | `generate_sequences()` |
| **HuggingFace** | `verl/workers/rollout/hf_rollout.py` | `HFRollout` | `generate_sequences()` |

**How to Change Inference:**
1. **Which inference engine to use?**
   - Config: `config.actor_rollout_ref.rollout.name` (set to `"sglang"`, `"vllm"`, or `"hf"`)
   - In script: `actor_rollout_ref.rollout.name=sglang`

2. **Modify generation parameters?**
   - Temperature, top-p, max tokens: `verl/workers/rollout/*/` generation parameter handling
   - For SGLang specifically: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~400` (sampling params)

3. **Add multi-turn support?**
   - SGLang: Already has multi-turn support via `verl/interactions/`
   - File: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~1800` (interaction handling)
   - Config: `config.data.chat_template` or `config.actor_rollout_ref.rollout.interactions`

4. **Add tool calling support?**
   - SGLang: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~2000` (tool parsing)
   - Config: `config.actor_rollout_ref.rollout.tools` or `config.tools` section

---

### 2. **TRAINING** (Where policy and value function are updated)

**Primary Entry Point:**
- `verl/trainer/ppo/ray_trainer.py:1090+` → Main training loop after rollout

**Source Code Locations:**

| Component | File | Key Function |
|---|---|---|
| **Main training loop** | `verl/trainer/ppo/ray_trainer.py:968+` | `RayPPOTrainer.fit()` |
| **Backward pass (actor)** | `verl/workers/actor/dp_actor.py` or `megatron_actor.py` | `compute_loss()` → `backward()` |
| **Backward pass (critic)** | `verl/workers/critic/dp_critic.py` or `megatron_critic.py` | `compute_loss()` → `backward()` |
| **Policy loss computation** | `verl/trainer/ppo/core_algos.py:~200+` | `@register_policy_loss()` decorated functions |
| **Advantage computation** | `verl/trainer/ppo/core_algos.py:~400+` | `@register_adv_est()` decorated functions |

**Training Loop Flow (in `fit()` method):**
```python
for epoch in range(total_epochs):
    for batch_dict in train_dataloader:
        batch = DataProto.from_single_dict(batch_dict)  # From ENV

        # STEP 1: INFERENCE
        gen_batch_output = self.actor_rollout_wg.generate_sequences(batch)

        # STEP 2: REWARD (ALGO)
        reward_tensor = self.reward_fn(gen_batch_output)

        # STEP 3: ADVANTAGE (ALGO)
        advantages = compute_advantages(...)  # GAE, GRPO, etc.

        # STEP 4: TRAINING
        actor_wg.step(loss=policy_loss)  # backward + optimizer step
        critic_wg.step(loss=value_loss)  # backward + optimizer step
```

**How to Change Training:**

1. **Change policy loss function?**
   - File: `verl/trainer/ppo/core_algos.py:50+`
   - How: Add `@register_policy_loss("my_loss")` decorator to custom function
   - Config: `config.algorithm.policy_loss_fn = "my_loss"`

2. **Change advantage estimator (GAE → GRPO)?**
   - File: `verl/trainer/ppo/core_algos.py:88+` (AdvantageEstimator enum)
   - How: Already implemented: GAE, GRPO, REINFORCE++, REMAX, RLOO, OPO, etc.
   - Config: `algorithm.adv_estimator=grpo` (in your script, already set)

3. **Change optimizer or learning rate?**
   - Config: `actor_rollout_ref.actor.optim.lr`, `actor_rollout_ref.actor.optim.beta1/2`, etc.
   - Source: `verl/workers/actor/dp_actor.py:~100` (optimizer creation)

4. **Add custom loss terms?**
   - File: `verl/trainer/ppo/core_algos.py:200+` (policy loss functions)
   - Add entropy loss, KL penalty: `core_algos.py:~250` (already has this)

---

### 3. **ENVIRONMENT/DATASET** (Where data comes from)

**Primary Entry Point:**
- `main_ppo.py:329` → `create_rl_dataset(config.data.train_files, ...)`

**Source Code Locations:**

| Component | File | Key Class |
|---|---|---|
| **Main dataset class** | `verl/utils/dataset/rl_dataset.py` | `RLHFDataset` |
| **Multi-turn dataset** | `verl/utils/dataset/multiturn_sft_dataset.py` | `MultiTurnSFTDataset` |
| **Dynamic data generation** | `verl/utils/dataset/dynamicgen_dataset.py` | `DynamicGenDataset` |
| **Data loading** | `main_ppo.py:369` | `create_rl_dataset()` |
| **Data configuration** | `verl/trainer/config/config.py` | Various dataclass configs |

**Dataset Loading Flow:**
```python
# In main_ppo.py:329
train_dataset = create_rl_dataset(
    data_paths=config.data.train_files,  # e.g., "~/data/gsm8k/train.parquet"
    data_config=config.data,
    tokenizer=tokenizer,
    processor=processor,
)

# Inside create_rl_dataset()
# 1. Check if custom_cls specified in config.data.custom_cls
# 2. Or use default: RLHFDataset
# 3. Load from parquet/json files
```

**How to Change Dataset:**

1. **Change data files?**
   - Config: `data.train_files`, `data.val_files`
   - In script: `data.train_files=$HOME/data/mydata/train.parquet`

2. **Use custom dataset class?**
   - Config: `data.custom_cls.path` and `data.custom_cls.name`
   - Example: `data.custom_cls.path=my_module.py data.custom_cls.name=MyDataset`
   - Must inherit from `torch.utils.data.Dataset`

3. **Use dynamic data generation (curriculum learning)?**
   - Config: `data.datagen.path` and `data.datagen.name`
   - Source: `verl/utils/dataset/dynamicgen_dataset.py:DynamicGenDataset`

4. **Change data format/tokenization?**
   - File: `verl/utils/dataset/rl_dataset.py:RLHFDataset.__init__()` and `__getitem__()`
   - Modify: prompt/response column names, tokenization logic, padding, etc.

5. **Use curriculum sampler?**
   - Config: `data.sampler.class_path` and `data.sampler.class_name`
   - Source: `verl/experimental/dataset/sampler.py:AbstractSampler`

**Key Dataset Config Parameters:**
```yaml
data:
  train_files: $HOME/data/gsm8k/train.parquet
  val_files: $HOME/data/gsm8k/test.parquet
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 1024
  truncation: 'error'
  shuffle: true
  seed: 42
  reward_fn_key: 'reward'  # Which column has rewards
  chat_template: null  # For multi-turn
```

---

### 4. **ALGORITHM** (Where rewards and advantages are computed)

**Primary Entry Points:**
- Reward: `main_ppo.py:317` → `load_reward_manager(config, tokenizer, ...)`
- Advantages: `ray_trainer.py:1090+` → advantage computation (inside fit loop)

**Source Code Locations:**

| Component | File | Key Function/Class |
|---|---|---|
| **Reward manager loading** | `verl/trainer/ppo/reward.py:118` | `load_reward_manager()` |
| **Custom reward functions** | `verl/trainer/ppo/reward.py:61` | `get_custom_reward_fn()` |
| **Reward computation** | `verl/trainer/ppo/reward.py:189` | `compute_reward()` |
| **Reward scoring (math)** | `verl/utils/reward_score/math_reward.py` | `compute_reward()` |
| **Reward scoring (GSM8K)** | `verl/utils/reward_score/gsm8k.py` | `compute_reward()` |
| **Naive reward manager** | `verl/workers/reward_manager/naive.py` | `NaiveRewardManager` |
| **Batch reward manager** | `verl/workers/reward_manager/batch.py` | `BatchRewardManager` |
| **DAPO reward manager** | `verl/workers/reward_manager/dapo.py` | `DAPORewardManager` |
| **Advantage estimators** | `verl/trainer/ppo/core_algos.py:200+` | `@register_adv_est()` functions |
| **Policy losses** | `verl/trainer/ppo/core_algos.py:50+` | `@register_policy_loss()` functions |

**Reward Computation Flow:**
```python
# In main_ppo.py:317
reward_fn = load_reward_manager(
    config=config,
    tokenizer=tokenizer,
    num_examine=0,
    **config.reward_model.get("reward_kwargs", {})
)

# load_reward_manager() does:
# 1. Check for custom reward function: get_custom_reward_fn()
# 2. If none, use default: default_compute_score() from reward_score/
# 3. Wrap in reward manager: NaiveRewardManager, BatchRewardManager, etc.
# 4. Return reward_fn callable

# Then in fit() loop (ray_trainer.py:1080+):
reward_tensor, reward_extra_info = self.reward_fn(gen_batch_output)
```

**How to Change Reward Function:**

1. **Use pre-built scoring function (math/GSM8K)?**
   - Config: `reward_model.reward_format` (e.g., "gsm8k", "math")
   - Source: `verl/utils/reward_score/`

2. **Write custom reward function?**
   - Create file: `my_reward.py`
   - Implement: `def compute_reward(data: DataProto, **kwargs) -> torch.Tensor`
   - Config:
     ```yaml
     reward_model:
       custom_reward_function:
         path: my_reward.py
         name: compute_reward
         reward_kwargs: {}
     ```

3. **Use sandbox execution (for code/test cases)?**
   - Config: `reward_model.sandbox_fusion.url` (sandbox service endpoint)
   - Source: `verl/utils/reward_score/sandbox_fusion/`

4. **Change reward manager type?**
   - Config: `reward_model.reward_manager` (naive, batch, dapo, prime, limited)
   - For rate-limiting: `verl/experimental/reward/reward_loop/limited.py`

**How to Change Advantage Estimator:**

1. **GAE (General Advantage Estimation):**
   - Source: `verl/trainer/ppo/core_algos.py:~500` (search for `@register_adv_est("gae")`)
   - Config: `algorithm.adv_estimator=gae`

2. **GRPO (Group Relative Policy Optimization):**
   - Source: `verl/trainer/ppo/core_algos.py:~600` (search for `@register_adv_est("grpo")`)
   - Config: `algorithm.adv_estimator=grpo` (your script uses this)

3. **Custom advantage estimator?**
   - File: `verl/trainer/ppo/core_algos.py`
   - Add: `@register_adv_est("my_advantage")` decorator
   - Config: `algorithm.adv_estimator=my_advantage`

---

## Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│ run_qwen3-8b.sh                                                          │
│ python3 -m verl.trainer.main_ppo [config overrides]                      │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ main_ppo.py:main()                                                        │
│ ├─ Hydra loads config from verl/trainer/config/ppo_trainer.yaml          │
│ └─ Merges with CLI overrides (e.g., algorithm.adv_estimator=grpo)        │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ main_ppo.py:run_ppo(config)                                               │
│ ├─ ray.init() - Initialize Ray cluster                                   │
│ └─ TaskRunner.run(config) - Remote task execution                        │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌─────────────────────┐   ┌──────────────────────┐
        │ ENVIRONMENT SETUP    │   │ ALGORITHM SETUP      │
        │                     │   │                      │
        │ create_rl_dataset() │   │ load_reward_manager()│
        │ ├─ RLHFDataset      │   │ ├─ custom_reward_fn()│
        │ ├─ Load parquet     │   │ ├─ reward_manager    │
        │ └─ Tokenize prompts │   │ └─ advantage est.    │
        └─────────────────────┘   └──────────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ RayPPOTrainer.__init__()                         │
        │ ├─ Initialize worker groups (actor, critic)      │
        │ ├─ Create rollout worker (vLLM/SGLang/HF)        │
        │ ├─ Setup distributed training backend (FSDP)     │
        │ └─ Setup data loaders                            │
        └──────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ RayPPOTrainer.init_workers()                     │
        │ ├─ Actor worker with policy model + optimizer    │
        │ ├─ Critic worker with value model + optimizer    │
        │ └─ Rollout worker with generation engine         │
        └──────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ RayPPOTrainer.fit() - MAIN TRAINING LOOP          │
        │ for epoch in total_epochs:                        │
        │   for batch in train_dataloader:                 │
        │                                                   │
        │ ┌─ STEP 1: INFERENCE ──────────────────────────┐ │
        │ │ actor_rollout_wg.generate_sequences(batch)   │ │
        │ │ └─> SGLang/vLLM/HF generates responses       │ │
        │ │     (multi-turn, tool calls if configured)   │ │
        │ └──────────────────────────────────────────────┘ │
        │                                                   │
        │ ┌─ STEP 2: REWARD COMPUTATION ─────────────────┐ │
        │ │ reward_tensor = reward_fn(gen_batch_output)  │ │
        │ │ └─> Custom fn or predefined (math/GSM8K)     │ │
        │ │     or sandbox execution                     │ │
        │ └──────────────────────────────────────────────┘ │
        │                                                   │
        │ ┌─ STEP 3: ADVANTAGE ESTIMATION ────────────────┐ │
        │ │ advantages = compute_advantages(             │ │
        │ │   rewards, values, masks                      │ │
        │ │   estimator=GAE/GRPO/etc                     │ │
        │ │ )                                             │ │
        │ └──────────────────────────────────────────────┘ │
        │                                                   │
        │ ┌─ STEP 4: POLICY & VALUE UPDATES ──────────────┐ │
        │ │ policy_loss = compute_policy_loss(           │ │
        │ │   old_logprobs, new_logprobs, advantages     │ │
        │ │ )                                             │ │
        │ │ actor_wg.backward(policy_loss)               │ │
        │ │ critic_wg.backward(value_loss)               │ │
        │ └──────────────────────────────────────────────┘ │
        │                                                   │
        │ Repeat for next batch...                         │
        └──────────────────────────────────────────────────┘
```

---

## Configuration Hierarchy

### Config Loading:
1. **Base config template:** `verl/trainer/config/ppo_trainer.yaml`
2. **Dataclass schema:** `verl/trainer/config/config.py` (CheckpointConfig, etc.)
3. **Worker configs:** `verl/workers/config/*.py` (ActorConfig, RolloutConfig, etc.)
4. **CLI overrides:** From `run_qwen3-8b.sh` arguments
   - Example: `algorithm.adv_estimator=grpo` overrides config file

### Key Config Paths (where to modify):

```yaml
# INFERENCE CONFIG
actor_rollout_ref:
  rollout:
    name: sglang              # Which engine: sglang, vllm, hf
    mode: async               # async or sync
    n: 5                       # num_return_sequences
    temperature: 1.0
    top_p: 0.9
    max_response_length: 1024
    log_prob_micro_batch_size_per_gpu: 32
    tensor_model_parallel_size: 2  # For tensor parallelism

# TRAINING CONFIG
actor_rollout_ref:
  actor:
    strategy: fsdp            # fsdp, fsdp2, megatron
    optim:
      lr: 1e-6
      beta1: 0.9
      beta2: 0.999
    use_kl_loss: True
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    entropy_coeff: 0
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 32

# ENVIRONMENT CONFIG
data:
  train_files: $HOME/data/gsm8k/train.parquet
  val_files: $HOME/data/gsm8k/test.parquet
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 1024
  custom_cls:
    path: null  # Custom dataset class
    name: null

# ALGORITHM CONFIG
algorithm:
  adv_estimator: grpo         # gae, grpo, reinforce_plus_plus, remax, rloo, opo
  policy_loss_fn: ppo         # ppo, grpo, etc.
  use_kl_in_reward: False
  kl_ctl:
    type: fixed               # fixed or adaptive
    init_kl_coef: 0.001

reward_model:
  enable: False               # Use learned reward model?
  reward_manager: naive       # naive, batch, dapo, prime, limited
  reward_format: null         # math, gsm8k, etc.
  custom_reward_function:
    path: null                # Path to custom reward fn
    name: null
  sandbox_fusion:
    url: null                 # Sandbox service endpoint
    max_concurrent: 64
```

---

## Where to Modify Each Component

### To Change Inference (add multi-turn, tool calls):
```
File: verl/workers/rollout/sglang_rollout/sglang_rollout.py
├─ generate_sequences() → Main generation loop
├─ Interaction handling → ~1800 (multi-turn support)
├─ Tool parsing → ~2000 (function calling)
└─ Sampling params → ~400

Config changes:
├─ actor_rollout_ref.rollout.name = "sglang"
├─ actor_rollout_ref.rollout.interactions = [...]
└─ actor_rollout_ref.rollout.tools = [...]
```

### To Change Training (policy loss, optimizer):
```
File: verl/trainer/ppo/core_algos.py
├─ @register_policy_loss() → Add custom loss
├─ @register_adv_est() → Add custom advantage
└─ Policy loss functions → ~200+

File: verl/workers/actor/dp_actor.py
├─ compute_loss() → Backward pass
└─ optimizer creation → ~100

Config changes:
├─ actor_rollout_ref.actor.optim.lr
├─ algorithm.adv_estimator = "my_adv"
└─ algorithm.policy_loss_fn = "my_loss"
```

### To Change Dataset/Environment:
```
File: verl/utils/dataset/rl_dataset.py
├─ RLHFDataset.__init__() → Data loading
└─ RLHFDataset.__getitem__() → Tokenization & formatting

Config changes:
├─ data.train_files = "path/to/data"
├─ data.custom_cls.path = "my_dataset.py"
└─ data.custom_cls.name = "MyDataset"
```

### To Change Reward (scoring, manager):
```
File: verl/trainer/ppo/reward.py
├─ get_custom_reward_fn() → Load custom reward
├─ load_reward_manager() → Choose reward manager
└─ compute_reward() → Reward computation

File: verl/utils/reward_score/gsm8k.py (or math_reward.py)
├─ compute_reward() → Scoring logic

File: verl/workers/reward_manager/naive.py (or other managers)
├─ __call__() → Reward management

Config changes:
├─ reward_model.reward_manager = "naive"
├─ reward_model.reward_format = "gsm8k"
├─ reward_model.custom_reward_function.path = "my_reward.py"
└─ reward_model.custom_reward_function.name = "compute_reward"
```

### To Change Advantage Estimation:
```
File: verl/trainer/ppo/core_algos.py
├─ AdvantageEstimator enum (line 88+) → Predefined estimators
├─ @register_adv_est("gae") → GAE implementation (~500)
├─ @register_adv_est("grpo") → GRPO implementation (~600)
└─ get_adv_estimator_fn() → Retrieve estimator

Config change:
└─ algorithm.adv_estimator = "grpo"  # or gae, reinforce_plus_plus, etc.
```

---

## Key Takeaways

1. **Entry Point:** `python3 -m verl.trainer.main_ppo` with Hydra config + CLI overrides

2. **Config System:** Hydra YAML + Dataclasses = type-safe, mergeable configs

3. **Four Components:**
   - **INFERENCE:** SGLang/vLLM/HF in `verl/workers/rollout/`
   - **TRAINING:** Actor/Critic backward pass in `verl/workers/actor/critic/`
   - **ENVIRONMENT:** Data loading in `verl/utils/dataset/`
   - **ALGORITHM:** Reward & advantage in `verl/trainer/ppo/` and `verl/utils/reward_score/`

4. **Main Loop:** `RayPPOTrainer.fit()` in `verl/trainer/ppo/ray_trainer.py:968+`

5. **To Modify:**
   - **Inference (multi-turn, tools):** Edit `sglang_rollout.py`, update config
   - **Training (loss, optimizer):** Register custom function in `core_algos.py`, update config
   - **Dataset:** Custom class in `rl_dataset.py` or external file + config
   - **Reward:** Custom function in external file + config in `reward.py`
   - **Advantage:** Register in `core_algos.py`, update config

---

## Example: How to Modify Reward Function

**Goal:** Change reward from GSM8K to custom math scoring

```python
# File: my_reward.py
import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    """Custom reward function for mathematical reasoning."""
    # data.responses contains the generated responses
    # Extract answers and compare to ground truth

    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.ground_truth_answers[i]

        # Your custom scoring logic
        if is_correct(response, ground_truth):
            rewards[i] = 1.0
        else:
            rewards[i] = 0.0

    return rewards

def is_correct(response, ground_truth):
    # Extract answer from response
    # Compare with ground_truth
    return extracted_answer == ground_truth
```

**Config:** In `run_qwen3-8b.sh` or YAML:
```yaml
reward_model:
  custom_reward_function:
    path: my_reward.py
    name: compute_reward
    reward_kwargs: {}
```

Or in script:
```bash
reward_model.custom_reward_function.path=my_reward.py \
reward_model.custom_reward_function.name=compute_reward
```

---

## Multi-Turn & Tool Call Support

**Multi-Turn:**
- Location: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~1800`
- Config: `actor_rollout_ref.rollout.interactions` or `data.chat_template`
- Setup interactions via `verl/interactions/` module

**Tool Calls:**
- Location: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:~2000`
- Config: `actor_rollout_ref.rollout.tools`
- Function call parser: SGLang's built-in `FunctionCallParser`

Both are already implemented in SGLang rollout; just configure them!
