"""
Core PPO Algorithm Functions.

Extracted from VERL's core_algos.py, adapted for standalone use.
Provides advantage estimation, policy loss computation, and KL penalty handling.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum


class AdvantageEstimator(str, Enum):
    """Advantage estimation method."""
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE = "reinforce"


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute masked mean."""
    masked = tensor * mask
    return masked.sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


def masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    shift_mean: bool = True,
) -> torch.Tensor:
    """Whiten (standardize) values while respecting mask."""
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    whitened = (values - mean) / torch.sqrt(var + 1e-8)
    return whitened * mask


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Adapted from HuggingFace TRL implementation.
    
    Args:
        token_level_rewards: Shape (batch_size, seq_len), rewards at each timestep
        values: Shape (batch_size, seq_len), value estimates
        response_mask: Shape (batch_size, seq_len), mask for valid tokens
        gamma: Discount factor
        lam: Lambda for GAE
        
    Returns:
        advantages: Shape (batch_size, seq_len)
        returns: Shape (batch_size, seq_len)
    """
    with torch.no_grad():
        next_values = 0.0
        last_gae_lam = 0.0
        advantages_reversed = []
        
        seq_len = token_level_rewards.shape[-1]
        
        for t in reversed(range(seq_len)):
            # TD residual
            delta = token_level_rewards[:, t] + gamma * next_values - values[:, t]
            
            # GAE
            last_gae_lam = delta + gamma * lam * last_gae_lam
            
            # Mask out padding tokens
            next_values = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * next_values
            last_gae_lam = last_gae_lam * response_mask[:, t] + (1 - response_mask[:, t]) * last_gae_lam
            
            advantages_reversed.append(last_gae_lam)
        
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        
        # Standardize advantages
        advantages = masked_whiten(advantages, response_mask)
    
    return advantages, returns


def compute_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO (Group Relative Policy Optimization) advantage.
    
    For outcome-based rewards (one scalar per response).
    
    Args:
        token_level_rewards: Shape (batch_size, seq_len)
        response_mask: Shape (batch_size, seq_len)
        index: Shape (batch_size,), group index for each sample
        epsilon: Numerical stability constant
        norm_adv: Whether to normalize by group standard deviation
        
    Returns:
        advantages: Shape (batch_size, seq_len)
        returns: Shape (batch_size, seq_len)
    """
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    
    id2score = {}
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        batch_size = scores.shape[0]
        
        # Group scores by index
        for i in range(batch_size):
            idx = index[i]
            if idx not in id2score:
                id2score[idx] = []
            id2score[idx].append(scores[i])
        
        # Compute statistics per group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                score_list = torch.stack(id2score[idx])
                id2mean[idx] = score_list.mean()
                id2std[idx] = score_list.std()
        
        # Normalize scores within groups
        for i in range(batch_size):
            idx = index[i]
            if norm_adv:
                scores[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[idx]
        
        # Broadcast to sequence length
        advantages = scores.unsqueeze(-1) * response_mask
        returns = advantages.clone()
    
    return advantages, returns


def compute_advantage(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    estimator: str = "gae",
    gamma: float = 0.99,
    lam: float = 0.95,
    index: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using specified estimator.
    
    Args:
        token_level_rewards: Token-level rewards
        values: Value function estimates
        response_mask: Valid token mask
        estimator: "gae" or "grpo"
        gamma: Discount factor
        lam: GAE lambda
        index: Group indices (required for GRPO)
        
    Returns:
        advantages, returns
    """
    if estimator == "gae":
        return compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif estimator == "grpo":
        if index is None:
            raise ValueError("index required for GRPO")
        return compute_grpo_advantage(
            token_level_rewards, response_mask, index
        )
    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO policy loss with clipping.
    
    Args:
        old_log_probs: Log probabilities from old policy, shape (batch_size, seq_len)
        log_probs: Log probabilities from current policy, shape (batch_size, seq_len)
        advantages: Advantage estimates, shape (batch_size, seq_len)
        response_mask: Valid token mask, shape (batch_size, seq_len)
        clip_ratio: PPO clip range
        loss_agg_mode: How to aggregate loss ("token-mean", "seq-mean", etc.)
        
    Returns:
        policy_loss: Scalar loss tensor
        metrics: Dictionary of metrics
    """
    # Compute probability ratio
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # Clipped objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    
    # Apply mask and aggregate
    if loss_agg_mode == "token-mean":
        policy_loss = (policy_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
    elif loss_agg_mode == "seq-mean":
        policy_loss = policy_loss.mean()
    else:
        policy_loss = policy_loss.mean()
    
    # Metrics
    with torch.no_grad():
        clipfrac = ((ratio - 1.0).abs() > clip_ratio).float()
        clipfrac = (clipfrac * response_mask).sum() / (response_mask.sum() + 1e-8)
        kl_div = masked_mean(-log_ratio, response_mask)
    
    metrics = {
        "policy_loss": policy_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "clipfrac": clipfrac.item(),
        "kl_div": kl_div.item(),
    }
    
    return policy_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO value function loss with clipping.
    
    Args:
        vpreds: Current value predictions, shape (batch_size, seq_len)
        returns: Target returns, shape (batch_size, seq_len)
        old_values: Old value estimates, shape (batch_size, seq_len)
        response_mask: Valid token mask, shape (batch_size, seq_len)
        cliprange_value: Clipping range for value loss
        loss_agg_mode: Loss aggregation mode
        
    Returns:
        value_loss: Scalar loss tensor
        metrics: Dictionary of metrics
    """
    # Clipped value loss
    vpreds_clipped = torch.clamp(
        vpreds,
        old_values - cliprange_value,
        old_values + cliprange_value
    )
    
    vf_loss_unclipped = (vpreds - returns) ** 2
    vf_loss_clipped = (vpreds_clipped - returns) ** 2
    vf_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)
    
    # Apply mask and aggregate
    if loss_agg_mode == "token-mean":
        vf_loss = (vf_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
    else:
        vf_loss = vf_loss.mean()
    
    vf_loss = 0.5 * vf_loss
    
    # Metrics
    with torch.no_grad():
        vf_clipfrac = (vf_loss_clipped > vf_loss_unclipped).float()
        vf_clipfrac = (vf_clipfrac * response_mask).sum() / (response_mask.sum() + 1e-8)
    
    metrics = {
        "value_loss": vf_loss.item(),
        "vf_clipfrac": vf_clipfrac.item(),
        "value_mean": vpreds.mean().item(),
        "return_mean": returns.mean().item(),
    }
    
    return vf_loss, metrics


def kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between two policies.
    
    Args:
        log_probs: Current policy log probs
        ref_log_probs: Reference policy log probs
        response_mask: Valid token mask
        
    Returns:
        kl_div: Scalar KL divergence
    """
    # KL(p_new || p_ref) = E[log p_new - log p_ref]
    kl = log_probs - ref_log_probs
    kl = torch.clamp(kl, min=-20, max=20)  # Numerical stability
    return masked_mean(-kl, response_mask)  # Negative because we want KL from ref to new


def apply_kl_penalty(
    token_level_rewards: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coef: float = 0.01,
) -> Tuple[torch.Tensor, float]:
    """
    Apply KL penalty to token-level rewards.
    
    Commonly used in instruction-following: reduces deviation from reference model.
    
    Args:
        token_level_rewards: Original rewards
        log_probs: Current policy log probs
        ref_log_probs: Reference policy log probs
        response_mask: Valid token mask
        kl_coef: KL penalty coefficient
        
    Returns:
        penalized_rewards: Rewards with KL penalty applied
        kl_penalty: Scalar KL divergence value
    """
    with torch.no_grad():
        # KL divergence (positive when p_new diverges from p_ref)
        kl_div = ref_log_probs - log_probs  # KL(ref || current)
        kl_div = torch.clamp(kl_div, min=-20, max=20)
        kl_div = kl_div * response_mask
        
        # Penalize rewards
        penalized_rewards = token_level_rewards - kl_coef * kl_div
        
        # Track average KL
        avg_kl = masked_mean(kl_div, response_mask)
    
    return penalized_rewards, avg_kl.item()


class KLController:
    """Adaptive KL penalty controller."""
    
    def __init__(
        self,
        init_kl_coef: float = 0.01,
        target_kl: float = 0.01,
        horizon: int = 10000,
        adaptive: bool = False,
    ):
        """
        Initialize KL controller.
        
        Args:
            init_kl_coef: Initial KL coefficient
            target_kl: Target KL divergence (for adaptive control)
            horizon: Number of steps over which to adapt
            adaptive: Whether to use adaptive control
        """
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
        self.adaptive = adaptive
    
    def update(self, current_kl: float, n_steps: int = 1):
        """Update KL coefficient based on current KL divergence."""
        if not self.adaptive:
            return
        
        # Proportional control
        error = current_kl / self.target_kl - 1.0
        error = np.clip(error, -0.2, 0.2)
        
        # Update with proportional term
        mult = 1.0 + error * n_steps / self.horizon
        self.value *= mult
        self.value = max(1e-6, self.value)  # Keep positive


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.
    
    Args:
        logits: Shape (..., vocab_size)
        
    Returns:
        entropy: Shape (...,)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
