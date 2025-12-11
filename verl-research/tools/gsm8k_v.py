"""
GSM8K helpers for rollout-only or training-like flows.

Provides:
- Dataset loader/builder (HF GSM8K)
- Final reward computation (strict GSM8K matching)
- Example interaction for a code-check follow-up (not wired by default)
"""

from typing import Any, Optional

from datasets import load_dataset
from verl.interactions.base import BaseInteraction
from verl.utils.reward_score import gsm8k as gsm8k_score
from verl.utils.dataset.rl_dataset import RLHFSequenceDataset


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_gsm8k_dataset(split: str = "train", limit: Optional[int] = None):
    ds = load_dataset("gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def build_dataset(
    tokenizer,
    max_prompt_len: int,
    max_response_len: int,
    split: str,
    limit: Optional[int],
):
    ds = load_gsm8k_dataset(split=split, limit=limit)
    return RLHFSequenceDataset(
        ds,
        tokenizer=tokenizer,
        processor=None,
        max_prompt_length=max_prompt_len,
        max_response_length=max_response_len,
        return_raw_chat=True,
    )


# -----------------------------------------------------------------------------
# Reward (final only)
# -----------------------------------------------------------------------------

def final_reward_gsm8k(answer_text: str, ground_truth: str) -> float:
    """
    Compute strict GSM8K reward at episode end.
    """
    return gsm8k_score.compute_score(
        answer_text,
        ground_truth,
        method="strict",
        format_score=0.0,
        score=1.0,
    )


# -----------------------------------------------------------------------------
# Interaction example (code-check follow-up)
# -----------------------------------------------------------------------------

class Gsm8kCodeCheckInteraction(BaseInteraction):
    """
    Example Interaction that asks the model to produce Python code to validate its answer.

    Flow:
    1) First assistant message: assumed to contain a numeric answer.
    2) Interaction responds with a prompt asking for Python code to verify the answer.
    3) After receiving the code, we terminate and compute final reward (strict GSM8K).

    Notes:
    - No code is executed here (safety). This is a structural example.
    - Final reward only; per-turn reward is 0.0.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances = {}
        self.max_turns = config.get("max_turns", 2)  # answer + code

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            from uuid import uuid4

            instance_id = str(uuid4())
        self._instances[instance_id] = {
            "turn": 0,
            "ground_truth": ground_truth or "",
            "answer_text": "",
            "got_code": False,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instances[instance_id]
        state["turn"] += 1

        # Grab last assistant message
        assistant_reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_reply = msg.get("content", "")
                break

        # First assistant turn: capture answer, ask for code
        if state["turn"] == 1:
            state["answer_text"] = assistant_reply
            feedback = (
                "Write a short Python snippet to verify your numeric answer. "
                "Print the parsed answer and the final numeric result. Do NOT describe—just code."
            )
            return False, feedback, 0.0, {"stage": "request_code"}

        # Second assistant turn: treat as code; terminate
        state["got_code"] = True
        feedback = "Received code. Stopping."
        return True, feedback, 0.0, {"stage": "code_received"}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        state = self._instances[instance_id]
        reward = final_reward_gsm8k(state["answer_text"], state["ground_truth"])
        state["final_reward"] = reward
        return reward

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instances:
            del self._instances[instance_id]
