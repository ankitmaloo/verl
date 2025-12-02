"""Generic environment that manages a batch of games."""

from tools import process_response_with_tools, format_tool_result


class Env:
    """
    Manages a batch of game instances.
    Completely generic - knows nothing about specific game types.
    Games implement their own parse_response() and format_observation().
    """

    def __init__(self, games: list):
        """
        Args:
            games: List of game instances (must implement BaseGame interface)
        """
        self.games = games
        self.size = len(games)
        self.convs = [[] for _ in range(self.size)]
        self.active = [True] * self.size
        self._init()

    def _init(self):
        for i, g in enumerate(self.games):
            prompt = g.get_initial_prompt()
            self.convs[i] = [{"role": "user", "content": prompt}]

    def get_prompts(self):
        """Returns conversation histories for active games, None for finished."""
        return [self.convs[i] if self.active[i] else None for i in range(self.size)]

    def step(self, responses):
        """
        Process model responses for all games.
        responses[i] should be None for finished games.
        """
        for i, resp in enumerate(responses):
            if not self.active[i] or resp is None:
                continue

            self.convs[i].append({"role": "assistant", "content": resp})

            # Check for tool calls first
            has_tool, tool_name, _, tool_result = process_response_with_tools(resp)
            if has_tool:
                self.convs[i].append(format_tool_result(tool_name, tool_result))
                continue

            # Delegate parsing to game
            action, give_up = self.games[i].parse_response(resp)
            result = self.games[i].step(action or "", give_up or False)

            if result.get("game_ended"):
                self.active[i] = False
                self.convs[i].append({
                    "role": "system",
                    "content": f"Game ended. Reward: {result.get('reward', 0)}"
                })
            else:
                obs = self.games[i].format_observation(result)
                self.convs[i].append({"role": "user", "content": obs})

    def done(self):
        """Returns True when all games are finished."""
        return not any(self.active)

    def get_batch(self):
        """Returns batch data for training."""
        return {
            'prompts': self.convs,
            'games': self.games,
        }
