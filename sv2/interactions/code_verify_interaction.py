"""
CodeVerifyInteraction: Asks the model to verify its answer with Python code.

After the model's first response, injects a user message asking it to write
Python code to verify the calculation. This creates a multi-turn flow:
  1. Model answers the question
  2. User (this interaction) says: "are you sure, can you write python code to verify"
  3. Model writes verification code
  4. Interaction terminates (or can continue for more turns)

Usage in interaction_config.yaml:
  interaction:
    - name: "code_verify"
      class_name: "sv2.interactions.code_verify_interaction.CodeVerifyInteraction"
      config:
        max_turns: 2
        feedback_message: "are you sure, can you write python code to verify the calculation"
"""

from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction


class CodeVerifyInteraction(BaseInteraction):
    """
    Interaction that requests code verification after the model's first answer.

    Config options:
        max_turns (int): Maximum interaction turns. Default 2 (answer + code).
        feedback_message (str): The user feedback to inject. Default asks for code verification.
        terminate_message (str): Message when terminating. Default "Thanks for the verification."
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances: dict[str, dict[str, Any]] = {}

        # Configurable feedback message
        self.max_turns = config.get("max_turns", 2)
        self.feedback_message = config.get(
            "feedback_message",
            "are you sure, can you write python code to verify the calculation"
        )
        self.terminate_message = config.get(
            "terminate_message",
            "Thanks for the verification."
        )

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Initialize interaction state for a trajectory."""
        if instance_id is None:
            instance_id = str(uuid4())

        self._instances[instance_id] = {
            "turn": 0,
            "ground_truth": ground_truth or "",
            "first_answer": "",
            "code_response": "",
            "messages_history": [],
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Generate user feedback based on current turn.

        Returns:
            (should_terminate, feedback_text, reward, metadata)
        """
        state = self._instances[instance_id]
        state["turn"] += 1
        state["messages_history"] = messages

        # Extract last assistant message
        assistant_reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_reply = msg.get("content", "")
                break

        # First turn: got initial answer, ask for code verification
        if state["turn"] == 1:
            state["first_answer"] = assistant_reply
            return (
                False,  # Don't terminate yet
                self.feedback_message,
                0.0,  # No reward yet (or could give partial reward)
                {"stage": "request_code", "turn": 1},
            )

        # Second turn (or later): got code response
        if state["turn"] >= self.max_turns:
            state["code_response"] = assistant_reply
            # Calculate final reward if ground_truth available
            reward = await self.calculate_score(instance_id)
            return (
                True,  # Terminate
                self.terminate_message,
                reward,
                {"stage": "code_received", "turn": state["turn"]},
            )

        # Intermediate turns (if max_turns > 2)
        return (
            False,
            "Please continue with the verification.",
            0.0,
            {"stage": "intermediate", "turn": state["turn"]},
        )

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """
        Calculate reward score.

        Override this for custom reward logic (e.g., execute code, check answer).
        Default returns 0.0 (reward handled elsewhere like reward_model).
        """
        # In a real implementation, you might:
        # 1. Parse the code from code_response
        # 2. Execute it safely in a sandbox
        # 3. Compare result to ground_truth
        # For now, return 0.0 and let external reward_model handle scoring
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up instance state."""
        if instance_id in self._instances:
            del self._instances[instance_id]
