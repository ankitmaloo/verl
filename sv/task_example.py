"""
Example task module.
- Supplies initial prompts
- Consumes responses to decide next prompts / termination
- Provides a reward function
Replace this with your task-specific logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def init_state(config: Dict[str, Any]) -> Dict[str, Any]:
    # Keep minimal state for multi-turn; extend as needed.
    return {"turn": 0, "history": []}


def get_initial_prompts(config: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    return ["You are a helpful assistant. Say hello and ask how you can assist."]


def process_responses(batch, config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    # Record decoded responses when available
    decoded = batch.non_tensor_batch.get("decoded_responses")
    if decoded is not None:
        state["history"].extend(decoded)

    state["turn"] += 1
    max_turns = config.get("actor_rollout_ref", {}).get("rollout", {}).get("multi_turn", {}).get(
        "max_assistant_turns", 1
    )

    # Dummy multi-turn: ask one follow-up then stop.
    if state["turn"] >= max_turns:
        return {"done": True}

    next_prompts = ["Thanks for your reply. Please provide one actionable next step."]
    return {"next_prompts": next_prompts, "done": False}


def build_reward_fn(tokenizer=None, config=None, state=None):
    # Simple reward: zero placeholder. Swap in your scorer.
    def reward_fn(batch, state=None) -> Tuple[torch.Tensor, dict]:
        responses = batch.batch["responses"]
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        return rewards, {}

    return reward_fn
