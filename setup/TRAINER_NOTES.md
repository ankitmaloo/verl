# Trainer Implementation Notes

## VERL's RayPPOTrainer Location

```
/Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/ray_trainer.py
```

Class: `RayPPOTrainer` (line 262+)

## Why Use VERL's Trainer

VERL's `RayPPOTrainer` is production-grade with:
- Distributed training across multiple GPUs/nodes
- Ray-based worker coordination
- Actor/Critic/Reward Model workers
- Support for FSDP, Megatron, vLLM, SGLang
- Checkpointing and metrics tracking
- Optimized data loading and batching

**Don't reimplement** - use it.

## Our Adapter Approach

Instead of reimplementing, `setup/trainer.py` provides:

1. **CustomEnvironmentDataset**: Converts environment → VERL dataset format
2. **EnvironmentRewardWrapper**: Wraps custom reward_fn → VERL reward interface
3. **CustomRayPPOTrainer**: Extends VERLRayPPOTrainer with:
   - Environment integration
   - Multi-turn inference support
   - Simplified initialization

## VERL's RayPPOTrainer Interface

### Initialization

```python
trainer = VERLRayPPOTrainer(
    config,                        # OmegaConf config
    tokenizer,                     # HuggingFace tokenizer
    role_worker_mapping,           # Dict[Role, WorkerType]
    resource_pool_manager,         # ResourcePoolManager
    reward_fn=None,                # Callable[[DataProto], Tensor]
    train_dataset=None,            # Optional[Dataset]
    val_dataset=None,              # Optional[Dataset]
    collate_fn=None,               # Optional[Callable]
    train_sampler=None,            # Optional[Sampler]
    device_name=None,              # Optional[str]
)
```

### Key Methods

```python
trainer.fit()                      # Main training loop
```

### Expected Inputs

**config** (OmegaConf):
```yaml
trainer:
  total_epochs: 10
  device: cuda
  project_name: my_project
  experiment_name: experiment_1
algorithm:
  gamma: 0.99
  lam: 0.95
  clip_ratio: 0.2
data:
  train_batch_size: 32
actor_rollout_ref:
  hybrid_engine: true
  rollout:
    n: 1
    temperature: 0.7
```

**tokenizer**:
- HuggingFace tokenizer with encode/decode

**role_worker_mapping**:
- Maps `Role` enum to worker implementations
- Roles: `ActorRollout`, `CriticValue`, `RefPolicy`, `RewardModel`

**resource_pool_manager**:
- Manages GPU allocation across workers
- Handles multi-node setup

**reward_fn**:
- Signature: `fn(data: DataProto) -> torch.Tensor`
- Input: batch from DataProto
- Output: rewards of shape `(batch_size,)`

**train_dataset**:
- Standard PyTorch Dataset
- `__getitem__` returns dict with `input_ids`, `attention_mask`, etc.

## What Our Adapter Does

### 1. CustomEnvironmentDataset
Converts environment state → text prompts → tokenized inputs

```python
# Environment generates state
obs = env.reset()

# We create prompts
prompt = f"Perform: {obs}"

# Tokenize for VERL
encoded = tokenizer(prompt, ...)

# Return in VERL format
return {"input_ids": ..., "attention_mask": ...}
```

### 2. EnvironmentRewardWrapper
Converts environment step → VERL reward format

```python
# VERL calls reward_fn(data)
def reward_fn(data: DataProto) -> torch.Tensor:
    # Extract response from data
    response = data.batch["responses"][i]
    
    # Execute in environment
    obs, reward, done, info = env.step(response)
    
    # Return tensor of rewards
    return torch.tensor([reward, ...])
```

### 3. CustomRayPPOTrainer
Extends VERL's trainer for our use case

```python
class CustomRayPPOTrainer(VERLRayPPOTrainer):
    def __init__(self, config, environment, ...):
        # Create dataset from environment
        dataset = CustomEnvironmentDataset(environment, tokenizer)
        
        # Wrap reward function
        reward_wrapper = EnvironmentRewardWrapper(environment, reward_fn)
        
        # Call VERL's trainer
        super().__init__(
            config=verl_config,
            tokenizer=tokenizer,
            reward_fn=reward_wrapper,
            train_dataset=dataset,
            ...
        )
    
    def fit(self):
        # Just call parent
        super().fit()
```

## Integration Points

### Where Custom Environment Hooks In

```
VERL's fit() loop
    ├─ Load batch from train_dataset
    │   └─ CustomEnvironmentDataset → text prompts
    │
    ├─ Actor generates responses
    │   └─ Multi-turn inference engine (optional)
    │
    ├─ Compute rewards
    │   └─ EnvironmentRewardWrapper calls env.step()
    │
    ├─ Compute advantages
    │   └─ VERL's core_algos functions
    │
    └─ Update policy
        └─ VERL's policy update
```

### Multi-Turn Integration

If you provide `multi_turn_engine`:

```python
trainer = CustomRayPPOTrainer(
    ...,
    multi_turn_engine=multi_turn,  # Optional
)

# During reward computation:
# Instead of single env.step(response)
# Can do: multi_turn.run_episode(env, response) for multi-turn
```

## TODO: Complete Implementation

The current adapter is a skeleton. To fully use VERL's trainer:

### 1. Create VERL Config

```python
def _create_verl_config(config: ConfigStore):
    # Map all fields from ConfigStore to OmegaConf
    # Required sections:
    # - trainer.* (epochs, device, logging)
    # - algorithm.* (gamma, lam, clip_ratio)
    # - model.* (model_name, dtype)
    # - data.* (batch_size, num_workers)
    # - actor_rollout_ref.* (rollout settings)
    pass
```

### 2. Setup Role Mapping

```python
def _default_role_mapping():
    # Need to implement based on VERL's:
    # - ActorRollout: inference worker
    # - CriticValue: value model worker
    # - RefPolicy: reference policy (optional)
    # - RewardModel: reward model (optional)
    pass
```

### 3. Setup Resource Pool

```python
def _default_resource_pool():
    # Create ResourcePoolManager with GPU allocation
    # Handles multi-GPU/multi-node setup
    pass
```

## Running Training

Once fully implemented:

```python
from setup import ConfigStore, create_trainer, SimpleGameEnvironment

config = ConfigStore()
config.training.num_epochs = 10

env = SimpleGameEnvironment(config.environment)

trainer = create_trainer(
    config=config,
    environment=env,
    tokenizer=tokenizer,
)

trainer.fit()
```

## Current Status

**What works:**
- Environment dataset creation
- Reward function wrapping
- Trainer class structure

**What needs implementation:**
- Full OmegaConf config conversion
- Role to worker mapping
- Resource pool setup
- Ray worker group initialization

## Next Steps

1. **Study VERL's config**: Look at `/verl/trainer/ppo/configs/`
2. **Understand role mapping**: Check `/verl/trainer/ppo/utils.py`
3. **Learn resource pools**: Check `/verl/single_controller/ray/`
4. **Implement full adapter**: Fill in the TODO sections
5. **Test on simple task**: Verify end-to-end flow

## Resources

- VERL Trainer: `verl/trainer/ppo/ray_trainer.py`
- Core algos: `verl/trainer/ppo/core_algos.py`
- Config schema: Look for OmegaConf dataclass definitions
- Examples: Check `recipe/` directory for complete examples
