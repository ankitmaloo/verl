"""Base classes for games and datasets."""


class BaseGame:
    """
    Base class for all game environments.
    Extend this and implement the required methods.
    """

    def step(self, action=None, give_up=False) -> dict:
        """
        Take an action in the game.

        Returns dict with:
        - First call (action=None): initial state with 'instructions', 'current_rule'
        - Subsequent calls: game state or {'game_ended': True, 'reward': float, ...}
        """
        raise NotImplementedError

    def parse_response(self, response: str) -> tuple:
        """
        Parse model response into (action, give_up).
        Override this for game-specific parsing.
        Default: return response as-is, give_up=False.
        """
        return response, False

    def get_initial_prompt(self) -> str:
        """Return the initial prompt for this game instance."""
        state = self.step()
        parts = []
        if state.get("instructions"):
            parts.append(state["instructions"])
        if state.get("current_rule"):
            parts.append(state["current_rule"])
        return "\n\n".join(parts) if parts else ""

    def format_observation(self, result: dict) -> str:
        """Format game result into observation string for the model."""
        if "all_rules" in result:
            return "Rules:\n" + "\n".join(
                f"{i+1}. {r}" for i, r in enumerate(result["all_rules"])
            )
        return str(result)

    @property
    def game_active(self) -> bool:
        raise NotImplementedError

    def calculate_reward(self, answer) -> float:
        raise NotImplementedError

    @property
    def attempts(self) -> list:
        """History of attempts/answers."""
        raise NotImplementedError


class GameDataset:
    """
    Base class for datasets that yield batches of game instances.
    Extend this for different data sources.
    """

    def get_batch(self, size: int) -> list:
        """Return a batch of `size` game instances."""
        raise NotImplementedError

    def __len__(self):
        """Total number of examples in dataset (or inf for generative)."""
        raise NotImplementedError

    def reset(self):
        """Reset iteration (for epoch boundaries)."""
        pass
