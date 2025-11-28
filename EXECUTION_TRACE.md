# Execution Trace: How run_qwen3-8b.sh Works Step-by-Step

## Phase 1: Script Execution → Config Loading

### Step 1: Run the script
```bash
bash examples/grpo_trainer/run_qwen3-8b.sh
```

**What happens:**
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    ...  # 40+ config overrides
```

---

### Step 2: Hydra loads config
- **File:** `verl/trainer/main_ppo.py:36`
- **Code:** `@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)`
- **What it does:**
  1. Loads base config: `verl/trainer/config/ppo_trainer.yaml`
  2. Merges with CLI overrides (e.g., `algorithm.adv_estimator=grpo`)
  3. Creates OmegaConf object with all parameters

**Config hierarchy:**
```
ppo_trainer.yaml (base)
  ├─ References actor/critic/rollout sub-configs
  ├─ Loads algorithm config
  └─ Loads data config
+ CLI overrides (run_qwen3-8b.sh arguments)
= Final merged config object
```

---

### Step 3: Hydra calls main()
- **File:** `verl/trainer/main_ppo.py:37`
- **Code:**
  ```python
  @hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
  def main(config):
      run_ppo(config)
  ```

---

## Phase 2: Ray Cluster Setup

### Step 4: ray.init()
- **File:** `verl/trainer/main_ppo.py:47-75`
- **What happens:**
  1. Checks if Ray already initialized: `ray.is_initialized()`
  2. Gets default Ray runtime env: `get_ppo_ray_runtime_env()`
  3. Merges with config: `config.ray_kwargs`
  4. Initializes Ray: `ray.init(**ray_init_kwargs)`

**Key config used:**
```
trainer.n_gpus_per_node: 8
trainer.nnodes: 1
ray_kwargs.ray_init: {...}
```

---

### Step 5: Create TaskRunner as Ray remote actor
- **File:** `verl/trainer/main_ppo.py:78`
- **Code:**
  ```python
  task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
  runner = task_runner_class.remote()
  ray.get(runner.run.remote(config))
  ```

**What this does:**
- Wraps `TaskRunner` class as a Ray actor
- Creates remote instance: `runner.remote()`
- Calls `run(config)` remotely

---

## Phase 3: TaskRunner Setup (main_ppo.py lines 106-366)

### Step 6: Add worker types
- **File:** `main_ppo.py:281-293`
- **Code:**
  ```python
  actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
  self.add_critic_worker(config)
  self.add_reward_model_worker(config)
  self.add_ref_policy_worker(config, actor_rollout_cls)
  ```

**What each does:**

| Function | Adds | Returns |
|---|---|---|
| `add_actor_rollout_worker()` | Actor + Rollout + Ref workers | Worker class, worker group class |
| `add_critic_worker()` | Critic (value function) worker | (none) |
| `add_reward_model_worker()` | Reward model (if enabled) | (none) |
| `add_ref_policy_worker()` | Reference policy for KL loss | (none) |

**Which worker to use depends on strategy:**
```python
# Determined by config.actor_rollout_ref.actor.strategy
if strategy in {"fsdp", "fsdp2"}:
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
elif strategy == "megatron":
    from verl.workers.megatron_workers import ActorRolloutRefWorker
```

**In your script:**
```
actor_rollout_ref.actor.strategy is NOT set (defaults to "fsdp")
→ Uses verl/workers/fsdp_workers.py:ActorRolloutRefWorker
```

---

### Step 7: Load dataset
- **File:** `main_ppo.py:329-344`
- **Code:**
  ```python
  train_dataset = create_rl_dataset(
      data_paths=config.data.train_files,         # ~/data/gsm8k/train.parquet
      data_config=config.data,
      tokenizer=tokenizer,
      processor=processor,
      is_train=True,
  )
  val_dataset = create_rl_dataset(
      data_paths=config.data.val_files,          # ~/data/gsm8k/test.parquet
      is_train=False,
  )
  ```

**Inside `create_rl_dataset()` (main_ppo.py:369-416):**
```python
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train, max_samples):
    # Check if custom dataset class specified
    if "custom_cls" in data_config and data_config.custom_cls.get("path"):
        dataset_cls = load_extern_type(...)  # Load from external file
    elif "datagen" in data_config and is_train:
        dataset_cls = DynamicGenDataset  # Dynamic curriculum learning
    else:
        dataset_cls = RLHFDataset  # Default: standard RLHF dataset

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )
    return dataset
```

**Your script uses:**
```
config.data.custom_cls is NOT set
config.data.datagen is NOT set
→ Uses RLHFDataset from verl/utils/dataset/rl_dataset.py
```

**Dataset source:** `verl/utils/dataset/rl_dataset.py:RLHFDataset`
- Loads parquet/JSON files
- Tokenizes prompts
- Returns (prompt, response) pairs

---

### Step 8: Load reward manager
- **File:** `main_ppo.py:317-322`
- **Code:**
  ```python
  reward_fn = load_reward_manager(
      config=config,
      tokenizer=tokenizer,
      num_examine=0,
      **config.reward_model.get("reward_kwargs", {})
  )
  val_reward_fn = load_reward_manager(
      config=config,
      tokenizer=tokenizer,
      num_examine=1,
      **config.reward_model.get("reward_kwargs", {})
  )
  ```

**Inside `load_reward_manager()` (verl/trainer/ppo/reward.py:118-186):**
```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    # Step 1: Try to load custom reward function
    compute_score = get_custom_reward_fn(config)  # Line 136

    # Step 2: If no custom function, use default
    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        if sandbox_config and sandbox_config.get("url"):
            # Use sandbox execution (code evaluation)
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=...)
        else:
            # Use default scoring (e.g., GSM8K, math rewards)
            final_compute_score = default_compute_score

    # Step 3: Wrap in reward manager
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Step 4: Instantiate reward manager
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
```

**In your script:**
```
config.reward_model.custom_reward_function is NOT set
→ Uses default_compute_score from verl/utils/reward_score/__init__.py

config.reward_model.reward_manager = "naive" (default)
→ Uses NaiveRewardManager from verl/workers/reward_manager/naive.py
```

**Source files:**
- Reward manager: `verl/workers/reward_manager/naive.py` (or batch.py, dapo.py, etc.)
- Default scoring: `verl/utils/reward_score/__init__.py:default_compute_score()`

---

### Step 9: Initialize RayPPOTrainer
- **File:** `main_ppo.py:348-361`
- **Code:**
  ```python
  trainer = RayPPOTrainer(
      config=config,
      tokenizer=tokenizer,
      processor=processor,
      role_worker_mapping=self.role_worker_mapping,      # Worker classes
      resource_pool_manager=resource_pool_manager,
      ray_worker_group_cls=ray_worker_group_cls,
      reward_fn=reward_fn,
      val_reward_fn=val_reward_fn,
      train_dataset=train_dataset,
      val_dataset=val_dataset,
      collate_fn=collate_fn,
      train_sampler=train_sampler,
  )
  ```

**Source:** `verl/trainer/ppo/ray_trainer.py:262`
- Stores all the components passed in
- Prepares for training

---

### Step 10: Initialize workers
- **File:** `main_ppo.py:363`
- **Code:**
  ```python
  trainer.init_workers()
  ```

**Inside `RayPPOTrainer.init_workers()` (ray_trainer.py:672+):**
- Creates actual worker instances (Actor, Critic, Rollout)
- Initializes models on GPUs
- Sets up distributed backend (FSDP)
- Creates data loaders

---

## Phase 4: Main Training Loop (ray_trainer.py:968+)

### Step 11: Start training
- **File:** `main_ppo.py:366`
- **Code:**
  ```python
  trainer.fit()
  ```

**Inside `RayPPOTrainer.fit()` (ray_trainer.py:968+):**

```python
def fit(self):
    # Line 979: Initialize logger
    logger = Tracking(...)

    # Line 989: Load checkpoint if resuming
    self._load_checkpoint()

    # Line 1008: Create progress bar
    for epoch in range(current_epoch, self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:  # Each batch from dataset

            # LINE 1034: Convert batch to DataProto
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # ═══════════════════════════════════════════════════════════════
            # STEP 1: INFERENCE (GENERATION)
            # ═══════════════════════════════════════════════════════════════
            # Line 1046: Repeat batch for N generations
            gen_batch_output = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,  # 5 in your script
                interleave=True
            )

            # Line 1055 OR 1057: ACTUAL GENERATION CALL
            if not self.async_rollout_mode:
                # Synchronous generation
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
            else:
                # Asynchronous generation
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

            # SOURCE: Calls into verl/workers/rollout/sglang_rollout/sglang_rollout.py:generate_sequences()
            # This is where SGLang/vLLM actually runs the model

            # ═══════════════════════════════════════════════════════════════
            # STEP 2: REWARD COMPUTATION
            # ═══════════════════════════════════════════════════════════════
            # Line ~1080-1090: Compute rewards
            reward_tensor, reward_extra_info = self.reward_fn(gen_batch_output)

            # SOURCE: Calls verl/trainer/ppo/reward.py:compute_reward()
            # Which calls the actual reward scoring function

            # ═══════════════════════════════════════════════════════════════
            # STEP 3: ADVANTAGE COMPUTATION
            # ═══════════════════════════════════════════════════════════════
            # Line ~1100-1120: Compute advantages
            advantages, returns = compute_advantages(
                reward_tensor=reward_tensor,
                values=critic_output,           # From critic network
                masks=response_mask,
                estimator=self.config.algorithm.adv_estimator,  # "grpo" in your script
            )

            # SOURCE: Calls verl/trainer/ppo/core_algos.py:get_adv_estimator_fn()
            # Which returns GRPO advantage computation function

            # ═══════════════════════════════════════════════════════════════
            # STEP 4: POLICY & VALUE UPDATES
            # ═══════════════════════════════════════════════════════════════
            # Line ~1130-1150: Compute losses
            policy_loss, loss_info = compute_policy_loss(
                old_log_probs=gen_batch_output.old_log_probs,
                log_probs=actor_output.log_probs,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode="token_level",
            )

            # SOURCE: Calls verl/trainer/ppo/core_algos.py:get_policy_loss_fn()

            # Line ~1160: Backward pass
            actor_wg.step(loss=policy_loss)    # Update policy

            # Line ~1170: Value function update
            critic_loss = compute_value_loss(...)
            critic_wg.step(loss=critic_loss)   # Update value function

            # SOURCE: Calls into verl/workers/actor/dp_actor.py or megatron_actor.py
            # and verl/workers/critic/dp_critic.py or megatron_critic.py

            # ═══════════════════════════════════════════════════════════════
            # LOGGING & CHECKPOINTING
            # ═══════════════════════════════════════════════════════════════
            # Line ~1180: Log metrics
            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': critic_loss.item(),
                'reward_mean': reward_tensor.mean().item(),
            }
            logger.log(data=metrics, step=self.global_steps)

            # Line ~1190: Save checkpoint
            if self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

            # Line ~1200: Run validation
            if self.global_steps % self.config.trainer.test_freq == 0:
                val_metrics = self._validate()

            self.global_steps += 1
```

---

## Detailed Inference Call Stack

### Where generation actually happens:

```
ray_trainer.py:1055
    └─> self.actor_rollout_wg.generate_sequences(gen_batch_output)
        (actor_rollout_wg is a RayWorkerGroup with SglangRollout or VllmRollout)

        └─> verl/workers/rollout/sglang_rollout/sglang_rollout.py:generate_sequences()
            ├─ Create sampling params (temperature, top_p, etc.)
            ├─ Prepare input (prompt tokenization)
            ├─ Call SGLang server with sampling params
            ├─ Get generated tokens
            ├─ Decode to text
            ├─ Handle multi-turn if configured
            ├─ Handle function calling if configured
            └─ Return DataProto with responses + log probs

            (Or for vLLM: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py)
            (Or for HF: verl/workers/rollout/hf_rollout.py)
```

**To add multi-turn or tool calls:**
- Location: `sglang_rollout.py:~1800` (multi-turn handling)
- Location: `sglang_rollout.py:~2000` (tool call parsing)
- Config: Set `config.actor_rollout_ref.rollout.interactions` or `config.actor_rollout_ref.rollout.tools`

---

## Detailed Reward Computation Call Stack

### Where reward is computed:

```
ray_trainer.py:1080
    └─> reward_tensor, reward_extra_info = self.reward_fn(gen_batch_output)
        (reward_fn is NaiveRewardManager or other manager)

        └─> verl/workers/reward_manager/naive.py:__call__()
            ├─ For each response in batch:
            │   └─> Call compute_score(response)
            │       └─> verl/utils/reward_score/__init__.py:default_compute_score()
            │           ├─ Check if sandbox (code execution) needed
            │           ├─ Extract answer from response
            │           ├─ Compare with ground truth
            │           └─ Return reward score

            └─> Return reward_tensor of shape [batch_size, seq_len]
```

**To change reward function:**
1. Write custom function in `my_reward.py`:
   ```python
   def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
       # Your custom logic
       return reward_tensor
   ```

2. Update config:
   ```yaml
   reward_model:
     custom_reward_function:
       path: my_reward.py
       name: compute_reward
   ```

3. Reload training: `python3 -m verl.trainer.main_ppo ... reward_model.custom_reward_function.path=my_reward.py`

---

## Detailed Advantage Computation Call Stack

### Where advantages are computed:

```
ray_trainer.py:1100
    └─> advantages, returns = compute_advantages(
            reward_tensor=reward_tensor,
            values=critic_values,
            masks=response_mask,
            estimator=self.config.algorithm.adv_estimator,  # "grpo"
        )

        └─> verl/trainer/ppo/core_algos.py:compute_advantages()
            ├─ Get estimator function by name:
            │   estimator_fn = get_adv_estimator_fn("grpo")
            │   └─> ADV_ESTIMATOR_REGISTRY["grpo"]
            │       └─> Points to @register_adv_est("grpo") decorated function
            │           └─> verl/trainer/ppo/core_algos.py:~600 (GRPO implementation)
            │
            └─> Call estimator_fn(
                    rewards=reward_tensor,
                    values=critic_values,
                    dones=masks,
                    ...
                )
                └─> Returns advantages, returns
```

**Advantage estimators available:**
```python
class AdvantageEstimator(Enum):
    GAE = "gae"                           # General Advantage Estimation
    GRPO = "grpo"                         # Group Relative Policy Optimization
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
```

**Your script uses:** `algorithm.adv_estimator=grpo`

---

## Detailed Policy Loss Computation Call Stack

### Where policy loss is computed:

```
ray_trainer.py:1130
    └─> policy_loss, loss_info = compute_policy_loss(
            old_log_probs=old_lp,
            log_probs=new_lp,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token_level",
        )

        └─> verl/trainer/ppo/core_algos.py:compute_policy_loss()
            ├─ Get policy loss function by name:
            │   loss_fn = get_policy_loss_fn(self.config.algorithm.policy_loss_fn)
            │   └─> POLICY_LOSS_REGISTRY["ppo"]
            │       └─> Points to @register_policy_loss("ppo") decorated function
            │           └─> verl/trainer/ppo/core_algos.py:~200 (PPO loss implementation)
            │
            └─> Call loss_fn(
                    old_log_probs=old_lp,
                    log_probs=new_lp,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode="token_level",
                )
                └─> Returns policy_loss, loss_info dict
```

**Policy loss functions available:**
```python
@register_policy_loss("ppo")
def ppo_loss(...):
    # PPO clipped objective
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1-eps, 1+eps) * advantages
    return -torch.min(surr1, surr2)

@register_policy_loss("grpo")
def grpo_loss(...):
    # GRPO loss
    ...

@register_policy_loss("reinforce")
def reinforce_loss(...):
    # REINFORCE loss
    ...
```

**Your script uses:** `algorithm.policy_loss_fn` (default is "ppo", can override)

---

## Summary: Where to Trace Each Component

| Component | Entry Point | Source File | Function | Line |
|---|---|---|---|---|
| **Inference** | `ray_trainer.py` | `1055` | `generate_sequences()` | Start: `sglang_rollout.py:~200` |
| **Reward** | `ray_trainer.py` | `1080` | `self.reward_fn()` | Start: `naive.py:__call__()` |
| **Advantage** | `ray_trainer.py` | `1100` | `compute_advantages()` | Start: `core_algos.py:~400` |
| **Policy Loss** | `ray_trainer.py` | `1130` | `compute_policy_loss()` | Start: `core_algos.py:~200` |
| **Actor Update** | `ray_trainer.py` | `1160` | `actor_wg.step()` | Start: `dp_actor.py:backward()` |
| **Critic Update** | `ray_trainer.py` | `1170` | `critic_wg.step()` | Start: `dp_critic.py:backward()` |

---

## Config Overrides Used in run_qwen3-8b.sh

```bash
# INFERENCE SETUP
actor_rollout_ref.rollout.name=vllm              # Use vLLM for inference
actor_rollout_ref.rollout.n=5                     # 5 return sequences per prompt
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # Tensor parallel

# TRAINING SETUP
actor_rollout_ref.actor.optim.lr=1e-6            # Learning rate
actor_rollout_ref.actor.ppo_mini_batch_size=256  # PPO batch size
actor_rollout_ref.actor.use_kl_loss=True         # KL penalty
actor_rollout_ref.actor.kl_loss_coef=0.001       # KL coefficient
actor_rollout_ref.actor.entropy_coeff=0          # No entropy bonus

# ALGORITHM SETUP
algorithm.adv_estimator=grpo                      # GRPO advantage estimation
algorithm.use_kl_in_reward=False                 # No KL in reward

# ENVIRONMENT SETUP
data.train_files=$HOME/data/gsm8k/train.parquet
data.val_files=$HOME/data/gsm8k/test.parquet
data.train_batch_size=1024
data.max_prompt_length=512
data.max_response_length=1024

# DISTRIBUTED TRAINING
trainer.n_gpus_per_node=8
trainer.nnodes=1
```

These overrides are merged with base config: `verl/trainer/config/ppo_trainer.yaml`
