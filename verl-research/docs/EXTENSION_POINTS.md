# Extension Points Guide

This guide shows you **where to make changes** for common algorithm modifications.

## 📍 Quick Reference

| I want to... | Extension | File | verl Component |
|--------------|-----------|------|----------------|
| Change how advantages are computed | `BaseAdvantageCompute` | `extensions/custom_advantages.py` | `verl/trainer/ppo/core_algos.py` |
| Add a new loss term | `BaseLossCompute` | `extensions/custom_losses.py` | `verl/workers/actor.py` |
| Shape/transform rewards | `BaseRewardShaper` | `extensions/custom_rewards.py` | `verl/trainer/ppo/reward.py` |
| Change sampling strategy | `BaseSampler` | `extensions/custom_samplers.py` | `verl/workers/rollout/*.py` |

---

## 1. Advantage Computation

**When to use**: You want to change how the model estimates which actions were good/bad.

### Common Use Cases

- **GRPO → DAPO**: Change from group-relative to direct alignment
- **PPO GAE tuning**: Modify λ and γ parameters
- **RLOO**: Use leave-one-out baseline
- **Custom baselines**: Use learned or hand-crafted baselines

### What verl Does (Baseline)

```python
# verl/trainer/ppo/core_algos.py

def compute_advantage(rewards, values, gamma=0.99, lam=0.95):
    """Standard GAE computation"""
    # Compute TD residuals
    deltas = rewards + gamma * values_next - values

    # Compute GAE
    advantages = compute_gae(deltas, gamma, lam)

    return advantages, returns
```

### Your Extension

File: `experiments/XX_your_variant/variant.py`

```python
from custom_advantages import BaseAdvantageCompute
import torch

class YourAdvantage(BaseAdvantageCompute):
    def compute(self, rewards, values=None, **kwargs):
        # YOUR ALGORITHM HERE
        advantages = rewards - rewards.mean()  # Example
        returns = rewards
        return advantages, returns
```

### Examples

| Paper/Idea | What Changes | Implementation |
|------------|--------------|----------------|
| DAPO | Advantage = reward - ref_reward | `DAPOAdvantage` (included) |
| GRPO | Advantage = reward - mean(group) | `GRPOAdvantage` (included) |
| RLOO | Leave-one-out baseline | `RLOOAdvantage` (included) |
| Custom | Your formula | Extend `BaseAdvantageCompute` |

---

## 2. Loss Functions

**When to use**: You want to add new terms to the training objective.

### Common Use Cases

- **Entropy bonus**: Encourage exploration
- **KL penalty**: Stay close to reference policy
- **Curiosity**: Add intrinsic rewards
- **Auxiliary tasks**: Multi-task learning

### What verl Does (Baseline)

```python
# verl/workers/actor.py

def compute_policy_loss(logprobs, old_logprobs, advantages, clip_range=0.2):
    """Standard PPO clipped loss"""
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantages
    loss = -torch.min(surr1, surr2).mean()
    return loss
```

### Your Extension

```python
from custom_losses import BaseLossCompute

class YourLoss(BaseLossCompute):
    def __init__(self, clip_range=0.2, your_coef=0.01):
        super().__init__()
        self.clip_range = clip_range
        self.your_coef = your_coef

    def compute(self, logprobs, old_logprobs, advantages, **kwargs):
        # Base PPO loss
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # YOUR TERM HERE
        your_term = self.compute_your_term(**kwargs)

        # Total
        total_loss = policy_loss + self.your_coef * your_term

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'your_term': your_term,
        }
```

### Examples

| Modification | Use Case | Implementation |
|--------------|----------|----------------|
| Entropy bonus | More exploration | `EntropyBonusLoss` (included) |
| KL penalty | Policy regularization | `KLPenaltyLoss` (included) |
| Curiosity | Intrinsic motivation | `CuriosityDrivenLoss` (template) |
| Value aux loss | Better critic | Add value loss term |

---

## 3. Reward Shaping

**When to use**: You want to transform rewards before they're used for training.

### Common Use Cases

- **Normalization**: Zero mean, unit variance
- **Clipping**: Prevent extreme values
- **Temporal smoothing**: Reduce noise
- **Length penalty**: Control response length
- **Custom bonuses**: Domain-specific rewards

### What verl Does (Baseline)

```python
# verl/trainer/ppo/reward.py

def shape_reward(raw_reward):
    """Identity - no shaping"""
    return raw_reward
```

### Your Extension

```python
from custom_rewards import BaseRewardShaper

class YourRewardShaper(BaseRewardShaper):
    def shape(self, rewards, **kwargs):
        # YOUR TRANSFORMATION HERE
        shaped = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return shaped
```

### Examples

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| Normalization | Stabilize training | `RewardNormalization` (included) |
| Clipping | Handle outliers | `RewardClipping` (included) |
| Temporal smoothing | Reduce variance | `TemporalSmoothing` (included) |
| Length penalty | Control length | `LengthPenalty` (included) |
| Composite | Chain multiple | `CompositeRewardShaper` (included) |

---

## 4. Sampling Strategy

**When to use**: You want to change how the model generates tokens during rollout.

### Common Use Cases

- **Temperature scheduling**: Anneal exploration
- **Top-p/nucleus sampling**: Quality control
- **Adaptive sampling**: Based on confidence
- **Beam search**: Better outputs

### What verl Does (Baseline)

```python
# verl/workers/rollout/vllm_rollout.py

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024
)
```

### Your Extension

```python
from custom_samplers import BaseSampler

class YourSampler(BaseSampler):
    def __init__(self, initial_temp=1.5, final_temp=0.7):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp

    def sample(self, logits, step=0, max_steps=100, **kwargs):
        # Compute current temperature
        progress = step / max_steps
        temp = self.initial_temp - progress * (self.initial_temp - self.final_temp)

        # Sample
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1)

        return samples
```

### Examples

| Strategy | Use Case | Implementation |
|----------|----------|----------------|
| Greedy | Deterministic | `GreedySampler` (included) |
| Temperature | Control randomness | `TemperatureSampler` (included) |
| Top-p | Nucleus sampling | `TopPSampler` (included) |
| Adaptive | Change over time | `AdaptiveTemperatureSampler` (included) |
| Confidence-based | Dynamic temperature | `ConfidenceBasedSampler` (included) |

---

## 🎯 Decision Tree: What Should I Change?

```
START: What aspect of the algorithm do you want to modify?

├─ "How good/bad were actions?"
│  └─> Advantage Computation
│      File: custom_advantages.py
│      Example: GRPO, DAPO, RLOO
│
├─ "What to optimize during training?"
│  └─> Loss Function
│      File: custom_losses.py
│      Example: Entropy, KL, Curiosity
│
├─ "How to transform rewards?"
│  └─> Reward Shaping
│      File: custom_rewards.py
│      Example: Normalize, clip, smooth
│
└─ "How to generate during rollout?"
   └─> Sampling Strategy
       File: custom_samplers.py
       Example: Temperature, top-p, adaptive
```

---

## 🔧 How to Apply Your Extension

### Step 1: Implement in variant.py

```python
# experiments/XX_your_variant/variant.py

from custom_advantages import BaseAdvantageCompute

class MyAdvantage(BaseAdvantageCompute):
    def compute(self, rewards, **kwargs):
        # Your algorithm
        return advantages, returns
```

### Step 2: Configure in config.yaml

```yaml
# experiments/XX_your_variant/config.yaml

variant:
  name: "my_variant"
  advantage_class: "variant.MyAdvantage"  # Point to your class
```

### Step 3: Tools apply it automatically

The training tools will:
1. Load your variant module
2. Instantiate your class
3. Monkey-patch verl to use it
4. Run training

---

## 📚 Real Paper Examples

### Example 1: DAPO (arxiv.org/abs/...)

**Paper claims**: Using reward difference as advantage is better than GAE.

**What to change**: Advantage computation

**Implementation**:
```python
class DAPOAdvantage(BaseAdvantageCompute):
    def compute(self, rewards, ref_rewards, **kwargs):
        advantages = rewards - ref_rewards
        returns = rewards
        return advantages, returns
```

**Config**:
```yaml
variant:
  advantage_class: "variant.DAPOAdvantage"
```

### Example 2: Entropy-Regularized RL

**Paper claims**: Adding entropy bonus improves exploration.

**What to change**: Loss function

**Implementation**:
```python
class EntropyLoss(BaseLossCompute):
    def __init__(self, entropy_coef=0.01):
        self.entropy_coef = entropy_coef

    def compute(self, logprobs, old_logprobs, advantages, **kwargs):
        # Base loss
        policy_loss = compute_ppo_loss(...)

        # Entropy bonus
        entropy = -logprobs.mean()
        entropy_loss = -self.entropy_coef * entropy

        total = policy_loss + entropy_loss
        return {'loss': total, 'entropy': entropy}
```

### Example 3: Temporal Credit Assignment

**Paper claims**: Smoothing rewards over time reduces variance.

**What to change**: Reward shaping

**Implementation**:
```python
class TemporalSmoothing(BaseRewardShaper):
    def shape(self, rewards, **kwargs):
        # Moving average
        smoothed = conv1d(rewards, kernel=gaussian_kernel)
        return smoothed
```

---

## ❓ FAQ

**Q: Can I change multiple things at once?**

Yes! Set multiple classes in config.yaml:

```yaml
variant:
  advantage_class: "variant.MyAdvantage"
  loss_class: "variant.MyLoss"
  reward_class: "variant.MyReward"
```

**Q: What if I need to change something not covered here?**

For deeper changes (e.g., new training loop, different optimizer), you might need to:
1. Create a custom trainer class
2. Modify the `train.py` script directly
3. Consult verl documentation for the specific component

**Q: How do I know my extension is actually being used?**

Add print statements or logging in your extension:

```python
def compute(self, rewards, **kwargs):
    print(f"🔧 Using MyAdvantage!")
    ...
```

**Q: Can I use existing verl implementations?**

Yes! Import from verl and call them:

```python
from verl.trainer.ppo import core_algos

class HybridAdvantage(BaseAdvantageCompute):
    def compute(self, rewards, values, **kwargs):
        # Use verl's GAE for part of it
        gae_adv, gae_ret = core_algos.compute_gae(...)

        # Mix with your idea
        my_adv = self.my_custom_logic(...)

        final_adv = 0.5 * gae_adv + 0.5 * my_adv
        return final_adv, gae_ret
```

---

## 🎓 Learning Path

1. **Start simple**: Try changing one thing (e.g., add entropy bonus)
2. **Compare with baseline**: Use tools to see if it helps
3. **Understand why**: Analyze the results
4. **Iterate**: Try variations
5. **Combine**: Mix successful extensions

---

## 🔗 See Also

- `QUICKSTART.md`: Step-by-step tutorial
- `EXAMPLES.md`: 5 complete example variants
- `../extensions/`: Full implementations
- `../experiments/template/`: Starting point

---

**Need help?** Check the included examples in `experiments/` or ask!
