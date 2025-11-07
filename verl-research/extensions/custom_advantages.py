"""
Custom advantage computation extensions.

Use this to implement new advantage estimators (GRPO variants, DAPO, etc.)
"""

import torch
from typing import Optional, Tuple
from .base import BaseExtension


class BaseAdvantageCompute(BaseExtension):
    """
    Base class for custom advantage computation.

    Override the `compute()` method to implement your algorithm.
    """

    def apply(self, *args, **kwargs):
        """Alias for compute()"""
        return self.compute(*args, **kwargs)

    def compute(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns.

        Args:
            rewards: Rewards tensor (batch_size, seq_len)
            values: Value estimates (batch_size, seq_len) - for GAE/PPO
            ref_logprobs: Reference log probs (batch_size, seq_len) - for KL penalty
            gamma: Discount factor
            lam: GAE lambda parameter
            **kwargs: Additional arguments

        Returns:
            advantages: Advantage estimates
            returns: Return estimates (for value function training)
        """
        raise NotImplementedError


class VanillaGAE(BaseAdvantageCompute):
    """
    Vanilla Generalized Advantage Estimation (for PPO).

    This is the default verl behavior - included as reference.
    """

    def compute(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard GAE computation"""

        batch_size, seq_len = rewards.shape

        # Compute TD residuals
        deltas = torch.zeros_like(rewards)
        for t in range(seq_len - 1):
            deltas[:, t] = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        deltas[:, -1] = rewards[:, -1] - values[:, -1]

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        advantages[:, -1] = deltas[:, -1]
        for t in reversed(range(seq_len - 1)):
            advantages[:, t] = deltas[:, t] + gamma * lam * advantages[:, t + 1]

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns


class GRPOAdvantage(BaseAdvantageCompute):
    """
    GRPO (Group Relative Policy Optimization) advantage.

    Uses relative ranking within sampled responses.
    """

    def compute(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int = 5,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GRPO advantage: Reward - mean(rewards in group)

        Args:
            rewards: (batch_size * num_samples, seq_len)
            num_samples_per_prompt: How many samples per prompt

        Returns:
            advantages: Centered advantages
            returns: Same as rewards (no critic)
        """

        # Reshape to (num_prompts, num_samples, seq_len)
        batch_size = rewards.shape[0]
        num_prompts = batch_size // num_samples_per_prompt
        rewards_grouped = rewards.reshape(num_prompts, num_samples_per_prompt, -1)

        # Compute mean reward per prompt
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)

        # Advantage = reward - mean
        advantages_grouped = rewards_grouped - mean_rewards

        # Reshape back
        advantages = advantages_grouped.reshape(batch_size, -1)
        returns = rewards  # No value function in GRPO

        return advantages, returns


class DAPOAdvantage(BaseAdvantageCompute):
    """
    DAPO (Direct Alignment via Preference Optimization) advantage.

    Example of a paper-inspired variant.
    Paper idea: Use reward - reference_reward as advantage.
    """

    def compute(
        self,
        rewards: torch.Tensor,
        ref_rewards: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DAPO advantage: reward - reference_reward

        Args:
            rewards: Current policy rewards
            ref_rewards: Reference policy rewards

        Returns:
            advantages: Reward difference
            returns: Current rewards
        """

        advantages = rewards - ref_rewards
        returns = rewards

        return advantages, returns


class RLOOAdvantage(BaseAdvantageCompute):
    """
    RLOO (REINFORCE Leave-One-Out) advantage.

    Uses leave-one-out baseline for variance reduction.
    """

    def compute(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int = 16,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RLOO: Use mean of other samples as baseline

        Args:
            rewards: (batch_size * num_samples, seq_len)
            num_samples_per_prompt: How many samples per prompt

        Returns:
            advantages: Leave-one-out advantages
            returns: Same as rewards
        """

        batch_size = rewards.shape[0]
        num_prompts = batch_size // num_samples_per_prompt

        # Reshape to (num_prompts, num_samples, seq_len)
        rewards_grouped = rewards.reshape(num_prompts, num_samples_per_prompt, -1)

        # For each sample, baseline = mean of OTHER samples
        sum_rewards = rewards_grouped.sum(dim=1, keepdim=True)
        baselines = (sum_rewards - rewards_grouped) / (num_samples_per_prompt - 1)

        # Advantage = reward - baseline
        advantages_grouped = rewards_grouped - baselines

        # Reshape back
        advantages = advantages_grouped.reshape(batch_size, -1)
        returns = rewards

        return advantages, returns


# Example: Custom advantage with your own idea
class CustomAdvantageTemplate(BaseAdvantageCompute):
    """
    Template for your custom advantage computation.

    Copy this and implement your idea!
    """

    def compute(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        YOUR ALGORITHM HERE

        Args:
            rewards: Rewards from environment
            values: Value estimates (if using critic)
            **kwargs: Other stuff you might need

        Returns:
            advantages: Your computed advantages
            returns: Target returns for value function
        """

        # TODO: Implement your algorithm
        # Example skeleton:

        # Step 1: Compute your baseline
        baseline = rewards.mean(dim=-1, keepdim=True)

        # Step 2: Compute advantages
        advantages = rewards - baseline

        # Step 3: Compute returns
        returns = rewards  # Or use a value function

        return advantages, returns
