# Complete Integration Example

Full working example integrating all components: environment, tools, inference, and training.

## Scenario: RL Agent for Text-Based Game with Tools

The agent plays a text-based game using:
- Environment for game state and rewards
- Tools for structured actions
- Multi-turn inference for planning
- PPO training to optimize policy

## Full Code

### 1. Define the Environment

```python
# game_env.py
from setup import BaseEnvironment, EnvironmentConfig
from typing import Tuple, Dict, Any

class TextAdventureEnvironment(BaseEnvironment):
    """Simple text adventure game environment."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.player_hp = 100
        self.inventory = []
        self.level = 0
        self.gold = 0
    
    def reset(self) -> str:
        """Reset game state."""
        self.player_hp = 100
        self.inventory = []
        self.level = 0
        self.gold = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self._done = False
        return self.get_state_description()
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action."""
        self.step_count += 1
        reward = 0.0
        info = {}
        
        action_lower = action.lower().strip()
        
        # Simulate different actions
        if "attack" in action_lower:
            if "slime" in action_lower:
                reward = 10.0
                self.gold += 5
                info["defeated_slime"] = True
            else:
                reward = -5.0
                self.player_hp -= 10
                info["missed_attack"] = True
        
        elif "heal" in action_lower:
            if "potion" in self.inventory:
                reward = 2.0
                self.player_hp = min(100, self.player_hp + 30)
                self.inventory.remove("potion")
                info["healed"] = True
            else:
                reward = -0.5
                info["no_potion"] = True
        
        elif "take" in action_lower or "pick" in action_lower:
            if "potion" in action_lower:
                self.inventory.append("potion")
                reward = 1.0
                info["took_potion"] = True
            else:
                reward = 0.5
                info["took_item"] = True
        
        elif "level up" in action_lower:
            if self.gold >= 10:
                reward = 20.0
                self.level += 1
                self.gold -= 10
                info["leveled_up"] = True
            else:
                reward = -1.0
                info["not_enough_gold"] = True
        
        else:
            reward = -0.1
            info["invalid_action"] = True
        
        # Update totals
        self.episode_reward += reward
        
        # Termination conditions
        if self.player_hp <= 0:
            self._done = True
            info["died"] = True
        elif self.level >= 3:
            self._done = True
            info["won"] = True
        elif self.step_count >= 10:
            self._done = True
        
        return self.get_state_description(), reward, self._done, info
    
    def get_state_description(self) -> str:
        """Get current game state as text."""
        return f"""
        === Game State ===
        HP: {self.player_hp}/100
        Level: {self.level}
        Gold: {self.gold}
        Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}
        Steps: {self.step_count}/10
        """.strip()
    
    def get_prompt_prefix(self) -> str:
        """System prompt for the model."""
        return """You are playing a text adventure game.
        
Available actions:
- attack <enemy>: Attack an enemy
- take <item>: Take an item
- heal: Use a potion to heal
- level up: Level up (costs 10 gold)

Your goal is to reach level 3 without dying.
Respond with ONE action at a time using the tool provided."""
```

### 2. Define Tools for Structured Actions

```python
# game_tools.py
from setup import Tool, ToolDefinition, ToolParameter

class AttackTool(Tool):
    def __init__(self, environment):
        self.env = environment
        super().__init__(ToolDefinition(
            name="attack",
            description="Attack an enemy",
            parameters=[
                ToolParameter(
                    name="enemy",
                    type="string",
                    description="Enemy to attack (e.g., slime, goblin)",
                    required=True,
                ),
            ],
        ))
    
    def execute(self, enemy: str, **kwargs) -> str:
        obs, reward, done, info = self.env.step(f"attack {enemy}")
        return f"Attacked {enemy}. Reward: {reward}. {obs}"

class TakeTool(Tool):
    def __init__(self, environment):
        self.env = environment
        super().__init__(ToolDefinition(
            name="take",
            description="Take an item",
            parameters=[
                ToolParameter(
                    name="item",
                    type="string",
                    description="Item to take",
                    required=True,
                ),
            ],
        ))
    
    def execute(self, item: str, **kwargs) -> str:
        obs, reward, done, info = self.env.step(f"take {item}")
        return f"Took {item}. Reward: {reward}. {obs}"

class HealTool(Tool):
    def __init__(self, environment):
        self.env = environment
        super().__init__(ToolDefinition(
            name="heal",
            description="Use a potion to heal",
            parameters=[],
        ))
    
    def execute(self, **kwargs) -> str:
        obs, reward, done, info = self.env.step("heal")
        return f"Heal attempt. Reward: {reward}. {obs}"

class LevelUpTool(Tool):
    def __init__(self, environment):
        self.env = environment
        super().__init__(ToolDefinition(
            name="level_up",
            description="Level up (costs 10 gold)",
            parameters=[],
        ))
    
    def execute(self, **kwargs) -> str:
        obs, reward, done, info = self.env.step("level up")
        return f"Level up attempt. Reward: {reward}. {obs}"
```

### 3. Setup Training Pipeline

```python
# main.py
import logging
from setup import (
    ConfigStore,
    RayPPOTrainer,
    create_inference_engine,
    MultiTurnInferenceEngine,
    JSONActionParser,
    MultiTurnManager,
    ToolRegistry,
)
from game_env import TextAdventureEnvironment
from game_tools import AttackTool, TakeTool, HealTool, LevelUpTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    config = ConfigStore()
    config.model.model_name = "meta-llama/Llama-2-7b-hf"
    config.inference.backend = "vllm"
    config.inference.max_tokens = 128
    config.training.learning_rate = 1e-5
    config.training.num_epochs = 5
    config.training.batch_size = 4
    config.checkpoint.save_interval = 2
    
    logger.info("Configuration initialized")
    
    # Initialize environment
    env = TextAdventureEnvironment(config.environment)
    logger.info("Environment initialized")
    
    # Initialize tools
    registry = ToolRegistry()
    registry.register(AttackTool(env))
    registry.register(TakeTool(env))
    registry.register(HealTool(env))
    registry.register(LevelUpTool(env))
    logger.info(f"Tools registered: {[t.definition.name for t in registry.list_definitions()]}")
    
    # Initialize multi-turn with tools
    parser = JSONActionParser(registry)
    tool_manager = MultiTurnManager(
        tool_registry=registry,
        action_parser=parser,
        environment=env,
        max_turns=5,
    )
    
    inference = create_inference_engine(
        model_path=config.model.model_name,
        backend=config.inference.backend,
    )
    
    multi_turn = MultiTurnInferenceEngine(
        engine=inference,
        max_turns=5,
        tool_manager=tool_manager,
        enable_tools=True,
    )
    logger.info("Multi-turn inference initialized")
    
    # Test one episode with tools
    logger.info("Running test episode with tools...")
    test_summary = multi_turn.run_episode(
        environment=env,
        initial_prompt=env.get_prompt_prefix(),
        max_tokens=128,
    )
    
    logger.info(f"Test episode: {test_summary['num_turns']} turns, "
                f"{test_summary['total_reward']} reward, "
                f"{test_summary['tools']['successful_tool_calls']} successful tool calls")
    
    # Reward function for RL
    def compute_episode_reward(episode_summary):
        """Compute RL reward from episode."""
        reward = episode_summary['total_reward']
        
        # Bonus for using tools effectively
        tools_bonus = episode_summary['tools']['successful_tool_calls'] * 2
        
        return reward + tools_bonus
    
    # Initialize trainer
    trainer = RayPPOTrainer(
        config=config,
        environment=env,
        reward_fn=compute_episode_reward,
    )
    logger.info("Trainer initialized")
    
    # Training loop
    logger.info("Starting training...")
    try:
        trainer.fit(num_epochs=config.training.num_epochs)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        trainer.cleanup()
        inference.shutdown()

if __name__ == "__main__":
    main()
```

### 4. Run Training

```bash
# Install dependencies
pip install torch transformers vllm

# Run training
python main.py
```

## Expected Output

```
INFO: Configuration initialized
INFO: Environment initialized
INFO: Tools registered: ['attack', 'take', 'heal', 'level_up']
INFO: Multi-turn inference initialized
INFO: Running test episode with tools...
INFO: Test episode: 3 turns, 15.5 reward, 2 successful tool calls
INFO: Trainer initialized
INFO: Starting training...
INFO: Epoch 1: Rollout phase
INFO: Epoch 1: Computing rewards
INFO: Epoch 1: Computing advantages
INFO: Epoch 1: Training phase
INFO: Epoch 1 metrics: {'total_loss': 0.245, 'policy_loss': 0.180, 'value_loss': 0.065, ...}
...
INFO: Training completed successfully!
```

## Monitoring

Track training with:

```python
# After training
for metric_dict in trainer.metrics_history:
    epoch = metric_dict['epoch']
    loss = metric_dict['total_loss']
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

## Customization Points

### Reward Function
Modify how RL rewards are computed:

```python
def compute_reward(episode_summary):
    # Custom reward computation
    base_reward = episode_summary['total_reward']
    
    # Penalties for inefficiency
    efficiency = episode_summary['num_turns'] / 10
    
    # Bonus for successful tool use
    tool_bonus = episode_summary['tools']['successful_tool_calls'] * 5
    
    return base_reward * efficiency + tool_bonus
```

### Tool Definition
Add more tools as needed:

```python
class InventoryTool(Tool):
    def __init__(self):
        super().__init__(ToolDefinition(
            name="inventory",
            description="Check inventory",
            parameters=[],
        ))
    
    def execute(self, **kwargs) -> str:
        return "Inventory content..."
```

### Action Parser
Use different parsing formats:

```python
# JSON format
from setup import JSONActionParser
parser = JSONActionParser(registry)

# ReAct format
from setup import ReActActionParser
parser = ReActActionParser(registry)

# Function calling format
from setup import FunctionCallingParser
parser = FunctionCallingParser(registry)
```

## Advanced: Multi-Agent

```python
# Run multiple environments in parallel
from concurrent.futures import ThreadPoolExecutor

def run_episode(env_id):
    env = TextAdventureEnvironment(config.environment)
    env.reset()
    # Run episode with tools...
    return episode_summary

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_episode, i) for i in range(4)]
    summaries = [f.result() for f in futures]
    
    # Aggregate results
    avg_reward = sum(s['total_reward'] for s in summaries) / len(summaries)
```

## Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
)

# Now you'll see:
# - Tool parsing details
# - Tool execution results
# - Environment state changes
# - Training metrics
```

## Next Steps

1. **Extend environment** with more game mechanics
2. **Add more tools** for richer action space
3. **Implement curriculum** - start easy, increase difficulty
4. **Collect offline data** - let untrained model explore
5. **Evaluate policy** - test on held-out episodes
6. **Scale to distributed** - use Ray for multi-GPU training
