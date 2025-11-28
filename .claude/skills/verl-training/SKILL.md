---
name: verl-training
description: Complete guide for building RL training pipelines with VERL. Write 4 simple Python files (algorithm.py, inference.py, training.py, env.py) that call VERL components directly for fast iteration. Covers: (1) Model rollout and generation (SGLang/vLLM/HuggingFace), (2) Policy and value function training, (3) Data pipeline and RL environments, (4) Reward computation and advantage estimation (GAE, GRPO, etc.), (5) Multi-turn conversations and tool calling, (6) Custom algorithms and loss functions. Use for standalone RL training scripts or modifying VERL internals.
---

# VERL Training

Write simple Python files that call VERL components for RL training. No abstraction, just direct usage of VERL's inference, training, and algorithm utilities.

**Best for:** Fast iteration, experimenting with datasets/RL environments, custom rewards, multi-turn conversations.

## Core Concepts

VERL (RL framework by ByteDance) organizes RL training into four interconnected components:

1. **Inference** - Model generation/rollout using SGLang, vLLM, or HuggingFace
2. **Training** - Policy and value function optimization with customizable loss functions
3. **Environment** - Data loading, sampling, and curriculum strategies
4. **Algorithm** - Reward computation and advantage estimation (GAE, GRPO, etc.)

The training loop follows this pattern:
```
Load Batch (ENV)
  → Generate Responses (INFERENCE)
    → Compute Rewards (ALGO)
      → Estimate Advantages (ALGO)
        → Update Policy (TRAINING)
          → Update Value Function (TRAINING)
```

## Quick Start: Write 4 Simple Python Files

**For fast iteration, write standalone scripts:**

1. `algorithm.py` - Main training loop (run this)
2. `inference.py` - Calls VERL rollout workers
3. `training.py` - Calls VERL actor/critic
4. `env.py` - Your data source (modify this for different tasks)

**See: [references/standalone.md](references/standalone.md)** for complete copy-paste patterns.

**Alternative:** Use VERL's built-in entry point `python3 -m verl.trainer.main_ppo` with Hydra config (see component guides below).

## The Four Core Components

### 1. Inference (Model Generation)

How to generate responses using different inference engines with support for multi-turn and tool calling.

**See: [references/inference.md](references/inference.md)** for comprehensive guide including:
- Selecting inference engine (SGLang, vLLM, HuggingFace)
- Enabling multi-turn conversations
- Adding function/tool calling
- Sampling parameter configuration
- Custom rollout implementations

**Quick config examples:**
```bash
# Use SGLang for best multi-turn support
actor_rollout_ref.rollout.name=sglang

# Enable multi-turn with chat template
data.chat_template=chatml

# Configure generation
actor_rollout_ref.rollout.temperature=0.7
actor_rollout_ref.rollout.top_p=0.9
actor_rollout_ref.rollout.n=5  # num_return_sequences
```

### 2. Training (Policy & Value Updates)

How to setup policy loss functions, value function training, and custom optimization strategies.

**See: [references/training.md](references/training.md)** for comprehensive guide including:
- Policy loss functions (PPO, GRPO, custom)
- Creating custom loss implementations
- Optimizer and learning rate configuration
- Distributed training strategies (FSDP, Megatron)
- Value function training
- Gradient and stability management

**Quick config examples:**
```bash
# Use GRPO advantage estimation
algorithm.adv_estimator=grpo

# Set learning rate
actor_rollout_ref.actor.optim.lr=1e-6

# Enable KL loss for stability
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001

# Use FSDP for distributed training
actor_rollout_ref.actor.strategy=fsdp
```

### 3. Environment (Dataset & Data Pipeline)

How to load data, create custom datasets, and implement curriculum learning strategies.

**See: [references/environment.md](references/environment.md)** for comprehensive guide including:
- Dataset loading (parquet, JSON, custom)
- Creating custom Dataset classes
- Data tokenization and formatting
- Curriculum learning samplers
- Batch size and sampling configuration
- Dynamic data generation

**Quick config examples:**
```bash
# Set data source
data.train_files=$HOME/data/gsm8k/train.parquet
data.val_files=$HOME/data/gsm8k/test.parquet

# Custom dataset
data.custom_cls.path=my_dataset.py
data.custom_cls.name=MyDataset

# Curriculum learning
data.sampler.class_path=my_sampler.py
data.sampler.class_name=CurriculumSampler
```

### 4. Algorithm (Reward & Advantage)

How to implement reward functions and advantage estimation strategies.

**See: [references/algorithms.md](references/algorithms.md)** for comprehensive guide including:
- Creating custom reward functions
- Reward managers and scaling
- Advantage estimators (GAE, GRPO, RLOO, etc.)
- Custom advantage implementations
- Sandbox-based reward computation for code tasks

**Quick config examples:**
```bash
# Custom reward function
reward_model.custom_reward_function.path=my_reward.py
reward_model.custom_reward_function.name=compute_reward

# Choose advantage estimator
algorithm.adv_estimator=grpo  # or gae, rloo, etc.

# Sandbox execution for code rewards
reward_model.sandbox_fusion.url=http://localhost:8000
```

## Execution Flow

The complete execution traces how config flows through the VERL pipeline:

1. **Hydra Config Loading** (`verl/trainer/main_ppo.py`)
   - Loads base config: `verl/trainer/config/ppo_trainer.yaml`
   - Merges with CLI overrides
   - Creates OmegaConf object

2. **Ray Cluster Setup** (`main_ppo.py:47-75`)
   - Initializes Ray for distributed training
   - Sets up runtime environment variables

3. **TaskRunner Setup** (`main_ppo.py:106-366`)
   - Initializes worker groups (Actor, Critic, Rollout, Reward)
   - Loads datasets
   - Creates reward managers
   - Initializes RayPPOTrainer

4. **Training Loop** (`verl/trainer/ppo/ray_trainer.py:968+`)
   - For each epoch/batch:
     - Generate sequences (INFERENCE)
     - Compute rewards (ALGORITHM)
     - Estimate advantages (ALGORITHM)
     - Update policy (TRAINING)
     - Update value function (TRAINING)

## Common Modifications

**For standalone scripts**, see **[references/standalone.md](references/standalone.md)** for:
- Switching inference engines (SGLang/vLLM/HF)
- Multi-turn conversations
- Tool calling
- Custom reward functions
- RL game environments
- Curriculum learning

**For modifying VERL internals**, see component-specific guides below.

## File Locations Reference

**Source Code Organization:**

```
Inference:
├─ SGLang: verl/workers/rollout/sglang_rollout/sglang_rollout.py
├─ vLLM: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
└─ HuggingFace: verl/workers/rollout/hf_rollout.py

Training:
├─ Losses: verl/trainer/ppo/core_algos.py (line 50+)
├─ Actor: verl/workers/actor/dp_actor.py
├─ Critic: verl/workers/critic/dp_critic.py
└─ Loop: verl/trainer/ppo/ray_trainer.py (line 968+)

Environment:
├─ Dataset: verl/utils/dataset/rl_dataset.py
├─ Sampler: verl/experimental/dataset/sampler.py
└─ Loader: verl/trainer/main_ppo.py (line 369+)

Algorithm:
├─ Reward: verl/trainer/ppo/reward.py (line 61+)
├─ Scoring: verl/utils/reward_score/
├─ Advantage: verl/trainer/ppo/core_algos.py (line 200+)
└─ Managers: verl/workers/reward_manager/
```

## Debugging Checklist

- **Generation not working**: Check `actor_rollout_ref.rollout.name`, model path, GPU memory
- **Reward not computed**: Verify custom function path, signature `(DataProto) -> Tensor`
- **Training diverges**: Check reward scale, advantage normalization, learning rate
- **Out of memory**: Reduce batch size, enable FSDP param offloading
- **Config not applied**: Check for typos, use `--cfg job` to inspect actual config

## Next Steps

1. **Start with references** - Read the component-specific guide for what you need to modify
2. **Use scripts** - Run helper scripts to validate your components in isolation
3. **Test incrementally** - Make one change at a time and monitor training
4. **Reference examples** - Check the scripts/ directory for implementation templates

---

**This skill provides:**
- **[references/standalone.md](references/standalone.md)** - **START HERE**: 4 simple Python files for fast iteration
- **[references/inference.md](references/inference.md)** - Inference engine selection and configuration
- **[references/training.md](references/training.md)** - Policy loss and training optimization
- **[references/environment.md](references/environment.md)** - Data pipeline and curriculum learning
- **[references/algorithms.md](references/algorithms.md)** - Reward and advantage functions
