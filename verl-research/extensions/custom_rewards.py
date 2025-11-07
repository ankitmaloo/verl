"""
Custom reward shaping extensions.

Use this to modify rewards (normalization, smoothing, bonuses, etc.)
"""

import torch
from typing import Dict, Any
from .base import BaseExtension


class BaseRewardShaper(BaseExtension):
    """
    Base class for reward shaping.

    Override shape() to implement your transformation.
    """

    def apply(self, *args, **kwargs):
        """Alias for shape()"""
        return self.shape(*args, **kwargs)

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Shape/transform rewards.

        Args:
            rewards: Raw rewards (batch_size, seq_len)
            **kwargs: Additional context (states, actions, etc.)

        Returns:
            shaped_rewards: Transformed rewards
        """
        raise NotImplementedError


class IdentityReward(BaseRewardShaper):
    """
    No shaping - return rewards as-is.

    This is the default behavior.
    """

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Return rewards unchanged"""
        return rewards


class RewardNormalization(BaseRewardShaper):
    """
    Normalize rewards to zero mean, unit variance.

    Common stabilization technique.
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(kwargs)
        self.eps = eps

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Normalize rewards"""

        mean = rewards.mean()
        std = rewards.std() + self.eps

        normalized = (rewards - mean) / std
        return normalized


class RewardClipping(BaseRewardShaper):
    """
    Clip rewards to a range.

    Prevents extreme reward values from dominating.
    """

    def __init__(
        self,
        min_value: float = -10.0,
        max_value: float = 10.0,
        **kwargs
    ):
        super().__init__(kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Clip rewards"""
        return torch.clamp(rewards, self.min_value, self.max_value)


class TemporalSmoothing(BaseRewardShaper):
    """
    Apply temporal smoothing to rewards.

    Example: Moving average over time steps.
    """

    def __init__(self, window_size: int = 3, **kwargs):
        super().__init__(kwargs)
        self.window_size = window_size

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Apply moving average smoothing"""

        batch_size, seq_len = rewards.shape
        smoothed = rewards.clone()

        for t in range(seq_len):
            start = max(0, t - self.window_size // 2)
            end = min(seq_len, t + self.window_size // 2 + 1)
            smoothed[:, t] = rewards[:, start:end].mean(dim=1)

        return smoothed


class LengthPenalty(BaseRewardShaper):
    """
    Add penalty/bonus based on sequence length.

    Useful for controlling response length.
    """

    def __init__(
        self,
        target_length: int = 100,
        penalty_coef: float = 0.1,
        **kwargs
    ):
        super().__init__(kwargs)
        self.target_length = target_length
        self.penalty_coef = penalty_coef

    def shape(
        self,
        rewards: torch.Tensor,
        lengths: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Add length penalty"""

        if lengths is None:
            # Assume all sequences are full length
            lengths = torch.full((rewards.shape[0],), rewards.shape[1])

        # Penalty = abs(length - target) * coef
        length_penalty = -self.penalty_coef * torch.abs(lengths.float() - self.target_length)

        # Add to rewards (broadcast across sequence)
        shaped_rewards = rewards + length_penalty.unsqueeze(-1)

        return shaped_rewards


class RewardScaling(BaseRewardShaper):
    """
    Scale rewards by a constant factor.

    Can help with learning stability.
    """

    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(kwargs)
        self.scale = scale

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Scale rewards"""
        return rewards * self.scale


class CompositeRewardShaper(BaseRewardShaper):
    """
    Apply multiple reward shapers in sequence.

    Example: Normalize → Clip → Smooth
    """

    def __init__(self, shapers: list, **kwargs):
        super().__init__(kwargs)
        self.shapers = shapers

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Apply all shapers in order"""

        shaped = rewards
        for shaper in self.shapers:
            shaped = shaper.shape(shaped, **kwargs)

        return shaped


class CustomRewardTemplate(BaseRewardShaper):
    """
    Template for your custom reward shaping.

    Copy and modify!
    """

    def __init__(self, your_param: float = 1.0, **kwargs):
        super().__init__(kwargs)
        self.your_param = your_param

    def shape(
        self,
        rewards: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        YOUR REWARD SHAPING HERE

        Args:
            rewards: Raw rewards
            **kwargs: Any additional context you need

        Returns:
            shaped_rewards: Your transformed rewards
        """

        # TODO: Implement your shaping logic
        # Example skeleton:

        # Step 1: Compute some statistics
        reward_mean = rewards.mean()

        # Step 2: Transform based on your idea
        shaped = rewards - reward_mean + self.your_param

        return shaped
