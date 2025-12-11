"""
Skeleton for a game-specific Interaction (multi-turn environment).

Purpose:
- Extend this class with your game logic (stateful environment, final reward).
- Plug it into a tool_agent_loop via interaction_config.yaml.

Notes:
- Keep final reward only (turn-level rewards can be logged but return 0).
- Store per-episode state in _instances keyed by instance_id.
- The agent loop will call:
  - start_interaction() once per trajectory
  - generate_response() after each assistant turn
  - calculate_score() at the end
  - finalize_interaction() to clean up
"""

from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction


class CustomGameInteraction(BaseInteraction):
    """
    Replace placeholders with your game logic.

    Config suggestions (put in interaction_config.yaml):
      interaction:
        - name: "my_game"
          class_name: "path.to.CustomGameInteraction"
          config:
            max_turns: 15
            # ... other game params
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances = {}
        self.max_turns = config.get("max_turns", 15)
        # Add more config-driven knobs as needed

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Initialize per-episode state.
        ground_truth can come from dataset (e.g., GSM8K answer).
        kwargs can carry extra scenario info.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        self._instances[instance_id] = {
            "turn": 0,
            "history": [],
            "ground_truth": ground_truth,
            "final_reward": 0.0,
            # add any other state you need
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Called after each assistant turn.

        Returns:
            should_terminate: bool
            response_content: str (becomes next user message)
            turn_reward: float (keep 0.0; final reward handled in calculate_score)
            additional_data: dict for logging
        """
        state = self._instances[instance_id]
        state["turn"] += 1

        # Extract last assistant reply
        assistant_reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_reply = msg.get("content", "")
                break
        state["history"].append(assistant_reply)

        # TODO: Evaluate assistant_reply against game rules
        # Example: score = self._evaluate_step(assistant_reply, state)
        score = 0.0  # placeholder

        # Decide if episode ends
        should_end = state["turn"] >= self.max_turns  # or custom success condition

        # Prepare feedback for next turn (user message)
        feedback = "Keep going."  # TODO: replace with game-specific feedback/hint

        additional = {
            "turn": state["turn"],
            "step_score": score,
        }
        return should_end, feedback, 0.0, additional  # keep turn reward zero

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """
        Compute final episode reward. Called once at the end.
        """
        state = self._instances[instance_id]
        # TODO: replace with your final scoring logic
        final_reward = state.get("final_reward", 0.0)
        return final_reward

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Cleanup state."""
        if instance_id in self._instances:
            del self._instances[instance_id]


# Helper stubs (optional): implement your game-specific helpers below
# def _evaluate_step(self, reply: str, state: dict) -> float:
#     ...
