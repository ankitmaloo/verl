"""
Custom loss function extensions.

Use this to add new loss terms (entropy bonus, curiosity, auxiliary losses, etc.)
"""

import torch
from typing import Dict, Any
from .base import BaseExtension


class BaseLossCompute(BaseExtension):
    """
    Base class for custom loss computation.

    Override compute() to add your loss terms.
    """

    def apply(self, *args, **kwargs):
        """Alias for compute()"""
        return self.compute(*args, **kwargs)

    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy loss.

        Args:
            logprobs: Current policy log probabilities
            old_logprobs: Old policy log probabilities (for PPO clipping)
            advantages: Advantage estimates
            **kwargs: Additional arguments

        Returns:
            Dict with:
                'loss': Total loss to minimize
                'policy_loss': Policy gradient loss
                'entropy': Entropy (for logging)
                ... any other terms you want to log
        """
        raise NotImplementedError


class VanillaPPOLoss(BaseLossCompute):
    """
    Standard PPO clipped loss.

    This is vanilla verl behavior - included as reference.
    """

    def __init__(self, clip_range: float = 0.2, **kwargs):
        super().__init__(kwargs)
        self.clip_range = clip_range

    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Standard PPO clipped loss"""

        # Probability ratio
        ratio = torch.exp(logprobs - old_logprobs)

        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages

        # Policy loss (negative because we maximize)
        policy_loss = -torch.min(surr1, surr2).mean()

        return {
            'loss': policy_loss,
            'policy_loss': policy_loss,
            'ratio_mean': ratio.mean(),
        }


class EntropyBonusLoss(BaseLossCompute):
    """
    PPO loss + entropy bonus.

    Encourages exploration by adding entropy term.
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        **kwargs
    ):
        super().__init__(kwargs)
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef

    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """PPO loss + entropy bonus"""

        # Standard PPO loss
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus (higher entropy = more exploration)
        if entropy is None:
            # Estimate entropy from logprobs
            entropy = -logprobs.mean()

        entropy_loss = -self.entropy_coef * entropy

        # Total loss
        total_loss = policy_loss + entropy_loss

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy,
        }


class KLPenaltyLoss(BaseLossCompute):
    """
    Policy loss + KL divergence penalty.

    Keeps policy close to reference policy.
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coef: float = 0.1,
        **kwargs
    ):
        super().__init__(kwargs)
        self.clip_range = clip_range
        self.kl_coef = kl_coef

    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Policy loss + KL penalty"""

        # Standard PPO loss
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence penalty
        kl = (logprobs - ref_logprobs).mean()
        kl_penalty = self.kl_coef * kl

        # Total loss
        total_loss = policy_loss + kl_penalty

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'kl_penalty': kl_penalty,
            'kl': kl,
        }


class CuriosityDrivenLoss(BaseLossCompute):
    """
    Policy loss + intrinsic curiosity bonus.

    Example of a more complex extension.
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        curiosity_coef: float = 0.01,
        **kwargs
    ):
        super().__init__(kwargs)
        self.clip_range = clip_range
        self.curiosity_coef = curiosity_coef

    def compute_curiosity(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute curiosity bonus.

        TODO: Implement your curiosity model here.
        This is a placeholder.
        """
        # Example: Use prediction error as curiosity
        # In practice, you'd train a forward model
        curiosity = torch.randn_like(states[:, 0])  # Placeholder
        return curiosity

    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Policy loss + curiosity bonus"""

        # Standard PPO loss
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Curiosity bonus
        if states is not None and actions is not None:
            curiosity = self.compute_curiosity(states, actions)
            curiosity_bonus = -self.curiosity_coef * curiosity.mean()  # Negative to add to loss
        else:
            curiosity_bonus = torch.tensor(0.0)

        # Total loss
        total_loss = policy_loss + curiosity_bonus

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'curiosity_bonus': curiosity_bonus,
        }


class CustomLossTemplate(BaseLossCompute):
    """
    Template for your custom loss.

    Copy and modify this!
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        your_coef: float = 0.01,
        **kwargs
    ):
        super().__init__(kwargs)
        self.clip_range = clip_range
        self.your_coef = your_coef

    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        YOUR LOSS COMPUTATION HERE

        Returns:
            Dict with at minimum:
                'loss': Total loss to minimize
                'policy_loss': Base policy loss
                ...any other terms for logging
        """

        # Base PPO loss
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # TODO: Add your custom term here
        your_term = torch.tensor(0.0)  # Replace with your computation

        # Total loss
        total_loss = policy_loss + self.your_coef * your_term

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'your_term': your_term,
        }
