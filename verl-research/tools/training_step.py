"""
Placeholder training step for rollout_only pipeline.

This does not update weights; it exists so the rollout driver can
invoke a “training” hook without edits when you later plug in an
optimizer.
"""

from typing import Any, Dict

from verl.protocol import DataProto


def training_step(rollout_out: DataProto, cfg: Any) -> Dict[str, Any]:
    """
    Stub training hook. Extend this to add optimizer updates.

    Args:
        rollout_out: DataProto from AgentLoopManager.generate_sequences
        cfg: TrainerConfig or raw config; passed through for future use

    Returns:
        Dict with placeholders/metrics.
    """
    metrics = {}
    # Example: surface reward stats if present
    rm_scores = rollout_out.batch.get("rm_scores") if rollout_out.batch is not None else None
    if rm_scores is not None:
        metrics["rm_scores_mean"] = rm_scores.mean().item()
        metrics["rm_scores_min"] = rm_scores.min().item()
        metrics["rm_scores_max"] = rm_scores.max().item()

    # TODO: plug in optimizer and loss computation here.
    metrics["train_step_ran"] = True
    return metrics
