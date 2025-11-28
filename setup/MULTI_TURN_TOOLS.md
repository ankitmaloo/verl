# Multi-Turn Inference with Tool Calling

Complete guide for multi-turn conversations with structured tool/action calling.

## Overview

The system supports:
- **Multi-turn conversations** with full history management
- **Tool/action calling** with structured JSON format
- **Multiple parsing formats**: JSON, ReAct, Function Calling
- **Tool execution and result feedback** into next turn
- **Environment integration** for RL tasks

## Architecture

```
┌──────────────────────────────────────┐
│  MultiTurnInferenceEngine            │
│  - Manages conversation history      │
│  - Orchestrates turns                │
├──────────────────────────────────────┤
│          Inference Engine            │
│  (VLLM or SGLang)                   │
│  - Generates model output            │
├──────────────────────────────────────┤
│       MultiTurnManager (Tools)       │
│  - Parses action/tool calls          │
│  - Executes tools                    │
│  - Returns results to conversation   │
├──────────────────────────────────────┤
│          Environment                 │
│  - Executes actions                  │
│  - Returns observations              │
└──────────────────────────────────────┘
```

## Quick Start

### 1. Define Tools

```python
from setup import Tool, ToolDefinition, ToolParameter, ToolRegistry

class QueryDatabaseTool(Tool):
    def __init__(self):
        definition = ToolDefinition(
            name="query_database",
            description="Query information from database",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="SQL query to execute",
                    required=True,
                ),
            ],
        )
        super().__init__(definition)
    
    def execute(self, query: str, **kwargs) -> str:
        """Execute database query."""
        # TODO: Implement your query logic
        results = self.query_database(query)
        return f"Query results: {results}"
    
    def query_database(self, query: str):
        # Implement your database access
        pass

# Register tools
registry = ToolRegistry()
registry.register(QueryDatabaseTool())
registry.register(SomeOtherTool())
```

### 2. Create Multi-Turn Manager

```python
from setup import JSONActionParser, MultiTurnManager

# Choose parser format
parser = JSONActionParser(registry)
# or: ReActActionParser(registry)
# or: FunctionCallingParser(registry)

manager = MultiTurnManager(
    tool_registry=registry,
    action_parser=parser,
    environment=my_env,
    max_turns=5,
)
```

### 3. Initialize Inference Engine with Tools

```python
from setup import create_inference_engine, MultiTurnInferenceEngine

# Create base inference engine
inference = create_inference_engine(
    model_path="meta-llama/Llama-2-7b-hf",
    backend="vllm",
)

# Wrap with multi-turn + tools
multi_turn = MultiTurnInferenceEngine(
    engine=inference,
    max_turns=5,
    tool_manager=manager,
    enable_tools=True,  # Enable tool calling
)
```

### 4. Run Episode

```python
episode_summary = multi_turn.run_episode(
    environment=my_env,
    initial_prompt="Solve this problem using available tools.",
    max_turns=5,
    max_tokens=256,
    temperature=0.7,
)

print(f"Turns: {episode_summary['num_turns']}")
print(f"Tools used: {episode_summary['tools']['tools_used']}")
print(f"Reward: {episode_summary['total_reward']}")
```

## Detailed Usage

### Tool Definition

```python
from setup import ToolDefinition, ToolParameter, Tool

class MyTool(Tool):
    def __init__(self):
        definition = ToolDefinition(
            name="my_tool",
            description="What this tool does",
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                ),
                ToolParameter(
                    name="param2",
                    type="number",
                    description="Second parameter",
                    required=False,
                    enum=["option1", "option2"],
                ),
            ],
        )
        super().__init__(definition)
    
    def execute(self, param1: str, param2: Optional[str] = None, **kwargs) -> str:
        """Execute the tool."""
        result = self.do_something(param1, param2)
        return result
    
    def do_something(self, param1, param2):
        # Implement logic
        pass
```

### Action Parsing Formats

**JSON Format** (recommended):
```python
from setup import JSONActionParser

parser = JSONActionParser(registry)

# Model output:
# {"tool": "query_database", "parameters": {"query": "SELECT * FROM users"}}

parsed = parser.parse(model_output)
# ParsedAction(
#   tool_name="query_database",
#   parameters={"query": "SELECT * FROM users"},
#   success=True,
# )
```

**ReAct Format**:
```python
from setup import ReActActionParser

parser = ReActActionParser(registry)

# Model output:
# Thought: I should query the database.
# Action: query_database
# Action Input: {"query": "SELECT * FROM users"}

parsed = parser.parse(model_output)
```

**Function Calling Format**:
```python
from setup import FunctionCallingParser

parser = FunctionCallingParser(registry)

# Model output:
# <function_calls>
# {"tool_name": "query_database", "parameters": {"query": "SELECT * FROM users"}}
# </function_calls>

parsed = parser.parse(model_output)
```

### Multi-Turn Flow

```python
from setup import MultiTurnManager, JSONActionParser, ToolRegistry

manager = MultiTurnManager(
    tool_registry=registry,
    action_parser=JSONActionParser(registry),
    environment=env,
    max_turns=5,
)

# Reset for new episode
manager.reset()

# Step 1: Generate and parse
model_output_1 = model.generate("Query the database...")
turn_1 = manager.step(model_output_1, environment_obs=current_obs)

# turn_1 contains:
# - turn_number: 1
# - model_output: raw model text
# - parsed_action: ParsedAction with tool info
# - tool_call: ToolCall with result (if tool executed)
# - observation: Result or environment observation

# Step 2: Get next context
next_obs = turn_1.observation

# Continue conversation...
for i in range(2, manager.max_turns + 1):
    model_output = model.generate(f"Based on: {next_obs}...")
    turn = manager.step(model_output, next_obs)
    next_obs = turn.observation
    
    if not manager.should_continue():
        break

# Get episode summary
summary = manager.get_episode_summary()
# {
#   "num_turns": 3,
#   "num_tool_calls": 2,
#   "successful_tool_calls": 2,
#   "tools_used": ["query_database", "execute_query"],
#   "turns": [...]
# }
```

### Tool Execution

Tools are executed automatically during `manager.step()`:

```python
# 1. Model generates output
model_output = '{"tool": "query_database", "parameters": {"query": "SELECT * FROM users"}}'

# 2. Manager parses the output
parsed = parser.parse(model_output)
# ParsedAction(
#   tool_name="query_database",
#   parameters={"query": "SELECT * FROM users"},
#   success=True,
# )

# 3. Manager executes the tool
tool = registry.get_tool("query_database")
result = tool.execute(**{"query": "SELECT * FROM users"})
# Returns: "Query results: [...]"

# 4. Result is wrapped in ToolCall
tool_call = ToolCall(
    tool_name="query_database",
    parameters={"query": "SELECT * FROM users"},
    result=result,
    success=True,
)

# 5. ToolCall is returned in Turn
turn = Turn(
    turn_number=1,
    model_output=model_output,
    parsed_action=parsed,
    tool_call=tool_call,
    observation=result,  # Tool result becomes observation
)
```

### MultiTurnInferenceEngine with Tools

```python
from setup import create_inference_engine, MultiTurnInferenceEngine

# Inference + Multi-turn + Tools integration
multi_turn = MultiTurnInferenceEngine(
    engine=create_inference_engine("meta-llama/Llama-2-7b-hf", backend="vllm"),
    max_turns=5,
    tool_manager=manager,
    enable_tools=True,
)

# Single step with tool processing
result = multi_turn.step(
    max_tokens=256,
    temperature=0.7,
    environment_obs=current_obs,
)
# result = {
#   "response": model output text,
#   "tool_call": ToolCall or None,
#   "tool_result": tool execution result or None,
# }

# Full episode with tools
summary = multi_turn.run_episode(
    environment=env,
    initial_prompt="Start task using tools.",
    max_turns=5,
    max_tokens=256,
    temperature=0.7,
)
# summary = {
#   "num_turns": 3,
#   "total_reward": 15.5,
#   "turns": [...],
#   "tools": {
#       "num_turns": 3,
#       "num_tool_calls": 2,
#       "successful_tool_calls": 2,
#       "tools_used": ["query_database"],
#       ...
#   },
#   ...
# }
```

## Example: Question Answering with Tools

```python
from setup import (
    Tool, ToolDefinition, ToolParameter, ToolRegistry,
    JSONActionParser, MultiTurnManager,
    create_inference_engine, MultiTurnInferenceEngine,
)

# Define tools
class SearchTool(Tool):
    def __init__(self):
        super().__init__(ToolDefinition(
            name="search",
            description="Search for information",
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
            ],
        ))
    
    def execute(self, query: str, **kwargs) -> str:
        # Simulate search
        return f"Search results for '{query}': [result1, result2, ...]"

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(ToolDefinition(
            name="calculate",
            description="Perform calculation",
            parameters=[
                ToolParameter("expression", "string", "Math expression", required=True),
            ],
        ))
    
    def execute(self, expression: str, **kwargs) -> str:
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# Setup
registry = ToolRegistry()
registry.register(SearchTool())
registry.register(CalculatorTool())

parser = JSONActionParser(registry)
manager = MultiTurnManager(registry, parser, max_turns=5)

# Inference
inference = create_inference_engine(
    "meta-llama/Llama-2-7b-hf",
    backend="vllm",
)

multi_turn = MultiTurnInferenceEngine(
    engine=inference,
    tool_manager=manager,
    enable_tools=True,
    max_turns=5,
)

# Run
question = "What is the capital of France? Then calculate 2+2."
summary = multi_turn.run_episode(
    environment=None,  # No environment needed for this task
    initial_prompt=question,
)

print(f"Turns: {summary['num_turns']}")
for turn in summary['turns']:
    print(f"  Turn {turn['turn']}: {turn['tool_call'].tool_name if turn['tool_call'] else 'no tool'}")
```

## Example: RL Task with Environment and Tools

```python
from setup import (
    BaseEnvironment, ToolRegistry, Tool, ToolDefinition, ToolParameter,
    JSONActionParser, MultiTurnManager,
    create_inference_engine, MultiTurnInferenceEngine,
)

class GameEnvironment(BaseEnvironment):
    def reset(self) -> str:
        self.state = "starting"
        return "You are in a game. Use tools to progress."
    
    def step(self, action: str) -> tuple:
        # Execute action and return observation, reward, done, info
        if "move" in action.lower():
            self.state = "moved"
            return "You moved.", 1.0, False, {}
        return "Invalid action.", -0.5, False, {}
    
    def get_state_description(self) -> str:
        return f"State: {self.state}"

class MoveTool(Tool):
    def __init__(self, env):
        self.env = env
        super().__init__(ToolDefinition(
            name="move",
            description="Move in the game",
            parameters=[
                ToolParameter("direction", "string", "Direction to move", required=True),
            ],
        ))
    
    def execute(self, direction: str, **kwargs) -> str:
        obs, reward, done, info = self.env.step(f"move {direction}")
        return f"Moved {direction}. Observation: {obs}"

# Setup
env = GameEnvironment()
registry = ToolRegistry()
registry.register(MoveTool(env))

parser = JSONActionParser(registry)
manager = MultiTurnManager(registry, parser, environment=env, max_turns=5)

# Inference
inference = create_inference_engine("meta-llama/Llama-2-7b-hf")
multi_turn = MultiTurnInferenceEngine(
    engine=inference,
    tool_manager=manager,
    enable_tools=True,
)

# Run RL episode
summary = multi_turn.run_episode(
    environment=env,
    initial_prompt="Play the game and win.",
    max_turns=5,
)

print(f"Total reward: {summary['total_reward']}")
print(f"Tools used: {summary['tools']['tools_used']}")
```

## Data Structures

### ParsedAction
```python
@dataclass
class ParsedAction:
    tool_name: str           # Name of tool to execute
    parameters: Dict[str, Any]  # Tool parameters
    raw_text: str            # Raw model output
    success: bool = True     # Whether parsing succeeded
    error: Optional[str] = None  # Error message if parsing failed
```

### ToolCall
```python
@dataclass
class ToolCall:
    tool_name: str           # Name of executed tool
    parameters: Dict[str, Any]  # Parameters passed to tool
    result: str              # Tool execution result
    success: bool            # Whether tool executed successfully
```

### Turn
```python
@dataclass
class Turn:
    turn_number: int         # Turn number (1-indexed)
    model_output: str        # Raw output from model
    parsed_action: Optional[ParsedAction]  # Parsed tool call
    tool_call: Optional[ToolCall]  # Executed tool info
    observation: str         # Observation for next turn
```

## Error Handling

Tool execution failures are handled gracefully:

```python
# If tool doesn't exist
success, result = registry.execute_tool("nonexistent_tool", param="value")
# success = False
# result = "Tool 'nonexistent_tool' not found"

# If tool execution raises exception
success, result = registry.execute_tool("calculator", expression="1/0")
# success = False
# result = "Tool execution error: division by zero"

# Parser failures
parsed = parser.parse("invalid json {{{")
# ParsedAction(
#   tool_name="",
#   parameters={},
#   success=False,
#   error="JSON parse error: ...",
# )
```

## Tips

1. **Always provide error feedback** to model by adding tool results/errors to conversation
2. **Use consistent JSON format** across all models - test different parsers
3. **Set reasonable max_turns** - multi-turn adds latency
4. **Log tool calls** for debugging and analysis
5. **Validate tool parameters** before execution
6. **Handle edge cases** in tool.execute() - don't let exceptions crash training
7. **Test tool parsing** with actual model outputs early

## Integration with Trainer

See `trainer.py` for how to integrate tools into RL training loop.
