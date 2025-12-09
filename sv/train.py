"""
Training hook placeholder: plug your value/policy loss here.
"""

from __future__ import annotations


def update_policy_stub(batch):
    """
    Stub for custom policy/value update.
    - batch.batch has: input_ids, attention_mask, responses, response_mask,
      token_level_scores (rewards), advantages, rollout_log_probs (if requested), etc.
    - Add your optimizer step here.
    """
    return {"status": "skipped"}
