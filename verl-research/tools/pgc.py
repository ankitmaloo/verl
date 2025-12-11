"""
Password Game (PG) task adapter for rollout_only.

Provides:
- build_dataset: construct a small synthetic dataset (default 16 instances) of PasswordGame prompts.
- final_reward_pg: compute end-of-episode reward.
- PasswordGameInteraction: interaction wrapper around exp/pg.PasswordGame.
"""

from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction
from verl.utils.dataset.rl_dataset import RLHFSequenceDataset

from exp.pg import PasswordGame, parse_resp


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def build_dataset(
    tokenizer,
    max_prompt_len: int,
    max_response_len: int,
    split: str,
    limit: Optional[int],
):
    """
    Build a small dataset of PasswordGame prompts.

    Args:
        tokenizer: HF tokenizer
        max_prompt_len: prompt length cap (passed through to RLHFSequenceDataset)
        max_response_len: response length cap
        split: unused (kept for interface compatibility)
        limit: number of instances; defaults to 16 if None
    """
    num_instances = limit or 16
    samples = []
    for i in range(num_instances):
        game = PasswordGame()
        prompt = game.get_initial_prompt()
        samples.append(
            {
                "raw_prompt": [{"role": "user", "content": prompt}],
                "game_seed": i,
            }
        )

    return RLHFSequenceDataset(
        samples,
        tokenizer=tokenizer,
        processor=None,
        max_prompt_length=max_prompt_len,
        max_response_length=max_response_len,
        return_raw_chat=True,
    )


# -----------------------------------------------------------------------------
# Reward (final only)
# -----------------------------------------------------------------------------

def final_reward_pg(game: PasswordGame, last_password: str) -> float:
    """End-of-episode reward using PasswordGame's calculate_reward."""
    if game is None:
        return 0.0
    return float(game.calculate_reward(last_password or ""))


# -----------------------------------------------------------------------------
# Interaction
# -----------------------------------------------------------------------------

class PasswordGameInteraction(BaseInteraction):
    """
    Wraps PasswordGame for multi-turn rollout.

    Protocol:
    - start_interaction: creates a fresh game instance
    - generate_response: consumes the latest assistant message as the password attempt,
      advances the game, and returns feedback (next rule/obs). Per-turn reward is 0.0.
    - calculate_score: computes final reward using the last submitted password.
    - finalize_interaction: cleans up state.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances = {}
        self.max_turns = config.get("max_turns", 16)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        game = PasswordGame()
        self._instances[instance_id] = {
            "game": game,
            "turn": 0,
            "last_password": "",
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
        game: PasswordGame = state["game"]

        # Latest assistant message as password attempt
        assistant_reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_reply = msg.get("content", "")
                break

        password, giveup = parse_resp(assistant_reply)
        state["last_password"] = password or assistant_reply or ""

        result = game.step(password=state["last_password"], give_up=bool(giveup))
        # Build feedback text from game state
        feedback = game.format_observation(result) if hasattr(game, "format_observation") else str(result)

        should_end = result.get("game_ended", False) or (state["turn"] >= self.max_turns)

        meta = {
            "turn": state["turn"],
            "gave_up": bool(giveup),
            "reward_snapshot": result.get("reward", None),
        }
        return should_end, feedback, 0.0, meta

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        state = self._instances.get(instance_id, {})
        game: PasswordGame = state.get("game")
        last_pw = state.get("last_password", "")
        return final_reward_pg(game, last_pw)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
