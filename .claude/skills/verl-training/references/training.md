# Training: Policy & Value Optimization

How to setup and customize policy loss functions, value function training, and optimization strategies.

## Table of Contents

1. Policy Loss Functions
2. Creating Custom Loss Functions
3. Advantage Estimators
4. Optimizer Configuration
5. Value Function Training
6. Distributed Training Strategies
7. Stability & Monitoring

## Policy Loss Functions

VERL provides multiple policy loss implementations via decorator-based registration:

### Available Loss Functions

**PPO (Proximal Policy Optimization)** - Default
```bash
algorithm.policy_loss_fn=ppo
```
Uses clipped objective to stabilize training:
```python
ratio = exp(log_probs - old_log_probs)
loss = -min(ratio * advantages, clamp(ratio, 1-ε, 1+ε) * advantages)
```

**GRPO (Group Relative Policy Optimization)**
```bash
algorithm.policy_loss_fn=grpo
```
Group-based relative advantage:
```python
# Compares rewards within groups to reduce variance
relative_rewards = rewards - group_mean(rewards)
```

**REINFORCE**
```bash
algorithm.policy_loss_fn=reinforce
```
Simple policy gradient:
```python
loss = -(log_probs * advantages)
```

**Custom Registered Losses**
```bash
# After registering with @register_policy_loss()
algorithm.policy_loss_fn=my_custom_loss
```

### Config Parameters for Policy Loss

```yaml
algorithm:
  policy_loss_fn: ppo  # Which loss function to use

actor_rollout_ref:
  actor:
    use_kl_loss: True  # Add KL penalty
    kl_loss_coef: 0.001  # KL penalty coefficient
    kl_loss_type: low_var_kl  # Type of KL estimation
    entropy_coeff: 0.01  # Optional entropy bonus
```

## Creating Custom Loss Functions

Add custom loss to `verl/trainer/ppo/core_algos.py`:

```python
from verl.trainer.ppo.core_algos import register_policy_loss
import torch
from typing import Optional, Dict, Any

@register_policy_loss("my_ppo_variant")
def my_custom_policy_loss(
    old_log_probs: torch.Tensor,  # [batch, seq_len]
    log_probs: torch.Tensor,       # [batch, seq_len]
    advantages: torch.Tensor,      # [batch, seq_len]
    response_mask: torch.Tensor,   # [batch, seq_len]
    loss_agg_mode: str = "token_level",
    config: Optional[Any] = None,
    rollout_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Custom PPO loss with modifications.

    Returns:
        loss: Scalar tensor
        loss_info: Dict with loss components for logging
    """
    # Compute probability ratio
    ratio = (log_probs - old_log_probs).exp()

    # PPO clipped objective (standard)
    epsilon = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)

    # YOUR CUSTOM MODIFICATIONS HERE:
    # Example: Add entropy bonus for exploration
    entropy_bonus = -log_probs.exp() * log_probs  # [batch, seq_len]
    entropy_weight = 0.01

    # Combine losses
    total_loss = policy_loss + entropy_weight * entropy_bonus

    # Apply response mask (only compute loss on valid tokens)
    total_loss = total_loss * response_mask

    # Aggregate
    if loss_agg_mode == "token_level":
        final_loss = total_loss.sum() / response_mask.sum().clamp(min=1)
    else:  # sample_level
        final_loss = (total_loss.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)).mean()

    # Return loss and diagnostics
    return final_loss, {
        "policy_loss": policy_loss.mean().item(),
        "entropy_bonus": entropy_bonus.mean().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "ratio_min": ratio.min().item(),
        "ratio_max": ratio.max().item(),
    }
```

### Using Custom Loss

```bash
# 1. Add function to core_algos.py with @register_policy_loss("my_ppo_variant")
# 2. Use in config:
algorithm.policy_loss_fn=my_ppo_variant
```

### Common Loss Modifications

**Add Entropy Regularization:**
```python
entropy = -log_probs.exp() * log_probs
total_loss = policy_loss + entropy_coeff * entropy
```

**Add KL Penalty:**
```python
kl_div = log_probs - old_log_probs
total_loss = policy_loss + kl_coeff * kl_div
```

**Reduce Outlier Impact:**
```python
# Clamp advantages to reduce impact of outliers
clamped_adv = torch.clamp(advantages, -k, k)
surr1 = ratio * clamped_adv
```

## Advantage Estimators

Different methods to estimate advantages from rewards and values:

### Available Estimators

**GAE (Generalized Advantage Estimation)** - Most common
```bash
algorithm.adv_estimator=gae
```
Combines TD and Monte Carlo estimates with lambda parameter.

**GRPO (Group Relative Policy Optimization)**
```bash
algorithm.adv_estimator=grpo
```
Uses relative group advantages (good for ranking).

**RLOO (Leave-One-Out)**
```bash
algorithm.adv_estimator=rloo
```
Variance reduction via leave-one-out sampling.

**REMAX**
```bash
algorithm.adv_estimator=remax
```
Relative expectation maximization.

**REINFORCE++**
```bash
algorithm.adv_estimator=reinforce_plus_plus
```
Simple policy gradient with return estimates.

**Custom Advantage Estimator**
```bash
algorithm.adv_estimator=my_advantage
```

### Creating Custom Advantage Estimator

Add to `verl/trainer/ppo/core_algos.py`:

```python
from verl.trainer.ppo.core_algos import register_adv_est
import torch

@register_adv_est("my_advantage")
def my_advantage_estimator(
    rewards: torch.Tensor,      # [batch, seq_len]
    values: torch.Tensor,       # [batch, seq_len]
    dones: torch.Tensor,        # [batch, seq_len]
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom advantage estimation.

    Returns:
        advantages: [batch, seq_len]
        returns: [batch, seq_len]
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Standard GAE implementation
    next_value = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]

        # TD error (temporal difference)
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]

        # Cumulative advantage with GAE weighting
        advantages[:, t] = delta + gamma * gae_lambda * (1 - dones[:, t]) * advantages[:, t + 1]

        # Return = advantage + value estimate
        returns[:, t] = advantages[:, t] + values[:, t]

    return advantages, returns
```

### GAE Configuration

```yaml
algorithm:
  adv_estimator: gae
  gae_lambda: 0.95  # Interpolation between TD (0) and MC (1)
  gamma: 0.99       # Discount factor
```

**Lambda parameter effects:**
- `lambda=0`: Pure TD, low bias high variance
- `lambda=0.95`: Balanced (recommended)
- `lambda=1.0`: Pure Monte Carlo, high bias low variance

## Optimizer Configuration

### Learning Rate

```yaml
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-6  # Learning rate
      beta1: 0.9  # Adam beta1
      beta2: 0.999  # Adam beta2
      weight_decay: 0.0  # L2 regularization
      eps: 1e-8  # Numerical stability
```

### Tuning Learning Rate

**Too High (training diverges):**
```bash
actor_rollout_ref.actor.optim.lr=1e-7  # Reduce
```

**Too Low (training too slow):**
```bash
actor_rollout_ref.actor.optim.lr=1e-5  # Increase
```

**Rule of thumb:**
- Start with 1e-6 for 7B models
- Scale with model size: 1e-5 for 70B, 1e-7 for 1B

### Optimizer Types

Default is AdamW. To use other optimizers, modify `verl/workers/actor/dp_actor.py:~80-120` where optimizer is created.

```python
# In dp_actor.py
if config.optim.type == "adamw":
    optimizer = torch.optim.AdamW(params, lr=config.optim.lr)
elif config.optim.type == "sgd":
    optimizer = torch.optim.SGD(params, lr=config.optim.lr)
```

## Value Function Training

The critic (value function) is trained alongside the policy:

### Critic Configuration

```yaml
critic:
  strategy: fsdp  # How critic is distributed (fsdp, megatron)
  # Critic has its own optim config under critic.optim
```

### Value Loss

Value function loss is computed as MSE:
```python
value_loss = ((returns - values) ** 2).mean()
```

The critic network produces value estimates used for:
1. Computing advantages
2. Bootstrapping returns for long sequences
3. Reducing variance in advantage estimation

**Location:** `verl/workers/critic/dp_critic.py`

### Training Dynamics

The policy and value function are updated in each training step:

```python
# In ray_trainer.py:1160+
# Policy update
actor_wg.step(loss=policy_loss)

# Value function update
critic_wg.step(loss=value_loss)
```

Both use the same rewards and advantages computed from the rollout.

## Distributed Training Strategies

Choose how to distribute actor and critic across GPUs:

### FSDP (Fully Sharded Data Parallel) - Recommended

```bash
actor_rollout_ref.actor.strategy=fsdp
critic.strategy=fsdp
```

**Config:**
```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      param_offload: False  # Offload params to CPU?
      optimizer_offload: False  # Offload optimizer to CPU?
      memory_limit_mb: null
```

**Best for:**
- Large models that don't fit on single GPU
- Training with multiple GPUs/nodes
- Memory efficiency via parameter sharding

### Megatron

```bash
actor_rollout_ref.actor.strategy=megatron
critic.strategy=megatron
```

**Best for:**
- Very large models (100B+)
- Advanced parallelism features
- Complex architectures

### Non-Distributed (Development)

```bash
actor_rollout_ref.actor.strategy=none
```

**Best for:**
- Development and testing
- Single GPU training with small models

## Stability & Monitoring

### Gradient Clipping

Control gradient explosion:

```yaml
actor_rollout_ref:
  actor:
    max_grad_norm: 1.0  # Clip gradients to this norm
```

### Loss Monitoring

Monitor these during training:

```python
metrics = {
    "policy_loss": policy_loss.item(),
    "value_loss": value_loss.item(),
    "ratio_mean": ratio.mean().item(),  # Should be ~1.0
    "ratio_std": ratio.std().item(),    # Should be small
    "advantage_mean": advantages.mean().item(),  # Should be ~0.0
    "advantage_std": advantages.std().item(),    # Should be ~1.0
    "learning_rate": current_lr,
    "kl_divergence": kl_div.mean().item(),  # Should be small
}
```

### Common Issues & Solutions

**Loss increases after first few steps:**
- Reduce learning rate by 10x
- Check reward scale (should be normalized)
- Verify advantage computation

**Training unstable (large loss spikes):**
- Enable KL loss: `use_kl_loss=True`
- Increase `kl_loss_coef`
- Reduce learning rate
- Check gradient norms

**Value function not training:**
- Increase critic learning rate
- Check critic network architecture
- Verify value targets are reasonable

## Entry Point in Training Loop

**File:** `verl/trainer/ppo/ray_trainer.py:1130-1170`

```python
# Compute policy loss
policy_loss, loss_info = compute_policy_loss(
    old_log_probs=gen_batch_output.old_log_probs,
    log_probs=actor_output.log_probs,
    advantages=advantages,
    response_mask=response_mask,
)

# Update policy
actor_wg.step(loss=policy_loss)

# Compute and update value function
value_loss = mse_loss(returns, critic_output.values)
critic_wg.step(loss=value_loss)
```

Both losses flow through their respective optimizer steps in the worker processes.
