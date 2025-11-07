"""
Custom sampling strategy extensions.

Use this to modify how the model samples tokens (temperature scheduling, top-p, etc.)
"""

import torch
from typing import Dict, Any
from .base import BaseExtension


class BaseSampler(BaseExtension):
    """
    Base class for custom sampling strategies.

    Override sample() to implement your sampling logic.
    """

    def apply(self, *args, **kwargs):
        """Alias for sample()"""
        return self.sample(*args, **kwargs)

    def sample(
        self,
        logits: torch.Tensor,
        step: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample tokens from logits.

        Args:
            logits: Model output logits (batch_size, vocab_size)
            step: Current generation step
            **kwargs: Additional context

        Returns:
            samples: Sampled token indices (batch_size,)
        """
        raise NotImplementedError


class GreedySampler(BaseSampler):
    """
    Always pick the highest probability token.

    Deterministic, no exploration.
    """

    def sample(
        self,
        logits: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Greedy sampling"""
        return logits.argmax(dim=-1)


class TemperatureSampler(BaseSampler):
    """
    Sample with fixed temperature.

    Standard sampling approach.
    """

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(kwargs)
        self.temperature = temperature

    def sample(
        self,
        logits: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Temperature-scaled sampling"""

        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Sample from distribution
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples


class TopPSampler(BaseSampler):
    """
    Nucleus (top-p) sampling.

    Only sample from tokens with cumulative probability > p.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ):
        super().__init__(kwargs)
        self.temperature = temperature
        self.top_p = top_p

    def sample(
        self,
        logits: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Top-p sampling"""

        # Scale by temperature
        scaled_logits = logits / self.temperature

        # Sort by probability
        probs = torch.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability > top_p
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Create mask in original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )

        # Set removed indices to very low probability
        filtered_logits = scaled_logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')

        # Sample from filtered distribution
        probs = torch.softmax(filtered_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples


class AdaptiveTemperatureSampler(BaseSampler):
    """
    Temperature that changes during generation.

    Example: Start high (explore) → End low (exploit)
    """

    def __init__(
        self,
        initial_temp: float = 1.5,
        final_temp: float = 0.7,
        max_steps: int = 100,
        **kwargs
    ):
        super().__init__(kwargs)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_steps = max_steps

    def get_temperature(self, step: int) -> float:
        """Compute temperature for current step"""

        progress = min(step / self.max_steps, 1.0)

        # Linear annealing
        temp = self.initial_temp - progress * (self.initial_temp - self.final_temp)

        return temp

    def sample(
        self,
        logits: torch.Tensor,
        step: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """Sample with adaptive temperature"""

        # Get current temperature
        temp = self.get_temperature(step)

        # Scale and sample
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples


class ConfidenceBasedSampler(BaseSampler):
    """
    Adjust temperature based on model confidence.

    High confidence → Lower temperature (more greedy)
    Low confidence → Higher temperature (more random)
    """

    def __init__(
        self,
        base_temp: float = 1.0,
        confidence_factor: float = 0.5,
        **kwargs
    ):
        super().__init__(kwargs)
        self.base_temp = base_temp
        self.confidence_factor = confidence_factor

    def sample(
        self,
        logits: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Sample with confidence-based temperature"""

        # Compute entropy as measure of uncertainty
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float))
        normalized_entropy = entropy / max_entropy

        # Temperature: high entropy → high temp, low entropy → low temp
        temp = self.base_temp * (1 + self.confidence_factor * normalized_entropy.unsqueeze(-1))

        # Sample with adjusted temperature
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples


class CustomSamplerTemplate(BaseSampler):
    """
    Template for your custom sampling strategy.

    Copy and modify!
    """

    def __init__(self, your_param: float = 1.0, **kwargs):
        super().__init__(kwargs)
        self.your_param = your_param

    def sample(
        self,
        logits: torch.Tensor,
        step: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        YOUR SAMPLING LOGIC HERE

        Args:
            logits: Model output (batch_size, vocab_size)
            step: Current generation step
            **kwargs: Any other context

        Returns:
            samples: Token indices (batch_size,)
        """

        # TODO: Implement your sampling strategy
        # Example skeleton:

        # Step 1: Transform logits based on your idea
        transformed_logits = logits * self.your_param

        # Step 2: Sample
        probs = torch.softmax(transformed_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples
