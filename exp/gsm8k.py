"""GSM8K game environment."""

from verl.utils.reward_score.gsm8k import compute_score, extract_solution
from base import BaseGame


def get_answer(example: dict) -> str:
    """Extract numeric answer from GSM8K example."""
    return example["answer"].split("####")[-1].strip().replace(",", "")


class GSM8KGame(BaseGame):
    """
    Single GSM8K problem as a game.
    Takes a problem dict from the dataset - does NOT do random selection.
    """

    def __init__(self, problem: dict):
        """
        Args:
            problem: Dict with 'question' and 'answer' from GSM8K dataset
        """
        self.question = problem["question"]
        self.ground_truth = get_answer(problem)
        self.answered = False
        self.correct = False
        self._attempts = []

    def step(self, answer=None, give_up=False):
        if answer is None:
            return {
                "instructions": "",
                "current_rule": self.question,
            }

        self._attempts.append(answer)

        if give_up:
            self.answered = True
            return {"game_ended": True, "reward": 0.0, "correct": False}

        score = compute_score(answer, self.ground_truth, method="flexible")
        self.correct = score == 1.0
        self.answered = True

        return {
            "game_ended": True,
            "reward": score,
            "correct": self.correct,
            "expected": self.ground_truth,
            "got": extract_solution(answer, method="flexible"),
        }

    def parse_response(self, response: str) -> tuple:
        """GSM8K: whole response is the answer."""
        return response, False

    def get_initial_prompt(self) -> str:
        """Just the question, no extra instructions for reasoning models."""
        return self.question

    def format_observation(self, result: dict) -> str:
        """GSM8K is single-turn, shouldn't be called."""
        return ""

    @property
    def game_active(self) -> bool:
        return not self.answered

    def calculate_reward(self, answer) -> float:
        return compute_score(answer, self.ground_truth, method="flexible")

    @property
    def attempts(self) -> list:
        return self._attempts

    # Alias for backward compat
    @property
    def password_history(self) -> list:
        return self._attempts
