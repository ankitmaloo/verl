# Adding Your Own Multi-Turn Environment to verl

A practical guide for creating custom multi-turn RL environments.

## Overview

verl supports two mechanisms for multi-turn environments:

| Mechanism | Use Case | Interface |
|-----------|----------|-----------|
| **Interaction** | Environment that provides feedback after each turn | `BaseInteraction` |
| **Tool** | Actions the model can call (with immediate rewards) | `BaseTool` |

**Use Interaction when**: You want the environment to evaluate the model's response and provide feedback (like a judge/teacher).

**Use Tool when**: You want the model to take specific actions that return observations (like calling an API, executing code, searching).

You can use both together: Tools for actions, Interaction for evaluation.

---

## Quick Start: 3 Files to Create

```
my_env/
├── my_interaction.py      # Your environment logic
├── my_interaction_config.yaml  # Registration
└── train_config.yaml      # Training setup
```

---

## Part 1: Creating an Interaction (Environment)

### Step 1: Implement Your Interaction

```python
# my_env/my_interaction.py

from typing import Any, Optional
from uuid import uuid4
from verl.interactions.base import BaseInteraction


class MyCustomInteraction(BaseInteraction):
    """
    Your custom multi-turn environment.

    The model generates responses, and this class:
    1. Evaluates the response
    2. Returns feedback for the next turn
    3. Decides when to terminate
    4. Computes rewards
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Instance state storage (one per trajectory)
        self._instances = {}

        # Your config params
        self.max_turns = config.get("max_turns", 5)
        self.success_threshold = config.get("success_threshold", 0.8)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,  # Passed from dataset
        **kwargs
    ) -> str:
        """
        Initialize state for a new trajectory.

        Called once at the start of each episode.
        kwargs contains data from your dataset's non_tensor_batch.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Initialize episode state
        self._instances[instance_id] = {
            "ground_truth": ground_truth,
            "turn_count": 0,
            "history": [],
            "best_score": 0.0,
        }

        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Process model's response and generate feedback.

        This is the CORE method - called after each model turn.

        Args:
            instance_id: Trajectory identifier
            messages: Full conversation history [{"role": "...", "content": "..."}]

        Returns:
            should_terminate: True to end episode
            response_content: Feedback message (becomes next user turn)
            turn_reward: Reward for this turn
            additional_data: Metrics/metadata for logging
        """
        state = self._instances[instance_id]
        state["turn_count"] += 1

        # Extract the model's latest response
        assistant_response = self._get_last_assistant_message(messages)
        state["history"].append(assistant_response)

        # ========================================
        # YOUR EVALUATION LOGIC HERE
        # ========================================
        score = self._evaluate_response(
            response=assistant_response,
            ground_truth=state["ground_truth"],
            turn=state["turn_count"]
        )

        # Track best score
        state["best_score"] = max(state["best_score"], score)

        # Determine termination
        should_terminate = (
            score >= self.success_threshold or  # Success!
            state["turn_count"] >= self.max_turns  # Out of turns
        )

        # Generate feedback for next turn
        if score >= self.success_threshold:
            feedback = "Correct! Great job."
        elif state["turn_count"] >= self.max_turns:
            feedback = f"Out of turns. The answer was: {state['ground_truth']}"
        else:
            feedback = self._generate_hint(state, score)

        # Compute reward
        # Option A: Sparse reward (only at end)
        # turn_reward = score if should_terminate else 0.0

        # Option B: Dense reward (every turn)
        turn_reward = score

        # Option C: Improvement-based reward
        # turn_reward = max(0, score - state.get("prev_score", 0))
        # state["prev_score"] = score

        additional_data = {
            "turn": state["turn_count"],
            "score": score,
            "best_score": state["best_score"],
        }

        return should_terminate, feedback, turn_reward, additional_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """
        Calculate final episode score.

        Called at the end of the episode for final reward computation.
        """
        state = self._instances[instance_id]
        return state["best_score"]

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Clean up episode state.

        Called when episode ends. Free resources here.
        """
        if instance_id in self._instances:
            del self._instances[instance_id]

    # ========================================
    # YOUR HELPER METHODS
    # ========================================

    def _get_last_assistant_message(self, messages: list[dict]) -> str:
        """Extract the last assistant response from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _evaluate_response(
        self,
        response: str,
        ground_truth: str,
        turn: int
    ) -> float:
        """
        YOUR EVALUATION LOGIC

        This is where your environment's semantics live.
        Return a score in [0, 1].
        """
        # Example: exact match
        if response.strip() == ground_truth.strip():
            return 1.0

        # Example: partial credit
        # return some_similarity_metric(response, ground_truth)

        return 0.0

    def _generate_hint(self, state: dict, score: float) -> str:
        """
        Generate feedback to help the model improve.

        This shapes the learning signal significantly!
        """
        turn = state["turn_count"]

        if score > 0.5:
            return "You're close! Check your final answer."
        elif turn == 1:
            return "That's not quite right. Think step by step."
        else:
            return "Still incorrect. Try a different approach."
```

### Step 2: Create Config File

```yaml
# my_env/my_interaction_config.yaml

interaction:
  - name: "my_env"  # Name used in logs
    class_name: "my_env.my_interaction.MyCustomInteraction"
    config:
      max_turns: 5
      success_threshold: 0.8
      # Add your custom config params here
```

### Step 3: Update Training Config

```yaml
# my_env/train_config.yaml

hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  max_prompt_length: 1024
  max_response_length: 1024
  train_batch_size: 256
  return_raw_chat: True  # IMPORTANT for multi-turn

  # Your dataset
  train_files: /path/to/train.parquet
  val_files: /path/to/val.parquet

actor_rollout_ref:
  hybrid_engine: True
  rollout:
    name: sglang
    multi_turn:
      enable: True
      max_user_turns: 5  # Max environment feedback turns
      interaction_config_path: my_env/my_interaction_config.yaml
```

---

## Part 2: Creating a Tool (Action)

Use this when you want the model to call specific functions.

### Step 1: Implement Your Tool

```python
# my_env/my_tool.py

from typing import Any, Optional
from uuid import uuid4
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse, OpenAIFunctionToolSchema


class MyCustomTool(BaseTool):
    """
    A tool the model can call during generation.

    Tools are actions with:
    - Defined input schema (OpenAI function format)
    - Execution logic
    - Immediate rewards
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

        # Your config
        self.api_url = config.get("api_url", "http://localhost:8000")

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Initialize tool state for a trajectory.

        Returns:
            instance_id: Unique ID for this trajectory
            initial_response: Optional initial message
        """
        if instance_id is None:
            instance_id = str(uuid4())

        self._instances[instance_id] = {
            "ground_truth": ground_truth,
            "call_count": 0,
            "results": [],
        }

        # Optional: return initial observation
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute the tool with given parameters.

        This is called when the model generates a tool call.

        Args:
            instance_id: Trajectory ID
            parameters: Parsed from model's tool call JSON

        Returns:
            response: ToolResponse with text/image/video
            reward: Immediate reward for this action
            metrics: Logging metadata
        """
        state = self._instances[instance_id]
        state["call_count"] += 1

        # ========================================
        # YOUR TOOL LOGIC HERE
        # ========================================

        # Example: Extract parameter
        query = parameters.get("query", "")

        # Example: Call external service
        result = await self._call_api(query)

        # Store result
        state["results"].append(result)

        # Compute immediate reward
        # Example: Reward for using tool correctly
        step_reward = 0.1 if result else -0.1

        # Return observation
        response = ToolResponse(
            text=f"Result: {result}",
            # image=...,  # Optional: include images
            # video=...,  # Optional: include video
        )

        metrics = {
            "call_count": state["call_count"],
            "query": query,
        }

        return response, step_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate final reward for this tool.

        Called at episode end.
        """
        state = self._instances[instance_id]

        # Example: Reward based on result quality
        if state["results"]:
            return self._evaluate_results(state["results"], state["ground_truth"])
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up tool state."""
        if instance_id in self._instances:
            del self._instances[instance_id]

    # ========================================
    # YOUR HELPER METHODS
    # ========================================

    async def _call_api(self, query: str) -> str:
        """Your API/service call logic."""
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(self.api_url, params={"q": query}) as resp:
        #         return await resp.text()
        return f"Mock result for: {query}"

    def _evaluate_results(self, results: list, ground_truth: str) -> float:
        """Evaluate tool usage quality."""
        # Your logic here
        return 1.0 if ground_truth in str(results) else 0.0
```

### Step 2: Create Tool Config

```yaml
# my_env/my_tool_config.yaml

tools:
  - class_name: "my_env.my_tool.MyCustomTool"
    config:
      type: native
      api_url: "http://localhost:8000"
    tool_schema:
      type: "function"
      function:
        name: "my_tool"
        description: "Describe what your tool does. The model sees this!"
        parameters:
          type: "object"
          properties:
            query:
              type: "string"
              description: "The search query"
            options:
              type: "object"
              description: "Optional settings"
              properties:
                limit:
                  type: "integer"
                  description: "Max results"
          required: ["query"]
```

### Step 3: Update Training Config

```yaml
# Add to your train_config.yaml

actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      max_assistant_turns: 10  # Max tool calls
      max_user_turns: 5        # Max environment turns
      tool_config_path: my_env/my_tool_config.yaml
      interaction_config_path: my_env/my_interaction_config.yaml  # Optional
```

---

## Part 3: Data Format

Your dataset needs specific fields for multi-turn:

```python
# Expected parquet columns:

# Required:
"prompt"           # Initial prompt/question
"data_source"      # Dataset name (for reward routing)

# For reward computation:
"reward_model": {
    "ground_truth": "the answer"  # Passed to interaction/tool
}

# Optional:
"extra_info"       # Additional metadata
```

Example data preparation:

```python
import pandas as pd

data = [
    {
        "prompt": [{"role": "user", "content": "Solve: 2 + 2 = ?"}],
        "data_source": "my_math_dataset",
        "reward_model": {"ground_truth": "4"},
    },
    # ...
]

df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

---

## Part 4: Complete Example - Code Execution Environment

Here's a full example of a code execution multi-turn environment:

```python
# code_env/code_interaction.py

import subprocess
import tempfile
from typing import Any, Optional
from uuid import uuid4
from verl.interactions.base import BaseInteraction


class CodeExecutionInteraction(BaseInteraction):
    """
    Environment where model writes code, we execute it,
    and provide feedback based on test results.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances = {}
        self.timeout = config.get("timeout", 5)
        self.language = config.get("language", "python")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,  # Expected output
        test_cases: Optional[list] = None,   # Input/output pairs
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        self._instances[instance_id] = {
            "ground_truth": ground_truth,
            "test_cases": test_cases or [],
            "attempts": [],
            "best_pass_rate": 0.0,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        state = self._instances[instance_id]

        # Extract code from model response
        code = self._extract_code(messages)
        state["attempts"].append(code)

        # Execute and test
        results = await self._run_tests(code, state["test_cases"])

        pass_rate = results["passed"] / max(len(state["test_cases"]), 1)
        state["best_pass_rate"] = max(state["best_pass_rate"], pass_rate)

        # Termination
        should_terminate = pass_rate == 1.0 or len(state["attempts"]) >= 5

        # Feedback
        if pass_rate == 1.0:
            feedback = "All tests passed! Great job!"
        else:
            feedback = self._format_test_feedback(results)

        # Reward: improvement-based
        prev_best = state.get("prev_pass_rate", 0)
        reward = max(0, pass_rate - prev_best) + (1.0 if pass_rate == 1.0 else 0)
        state["prev_pass_rate"] = pass_rate

        return should_terminate, feedback, reward, {
            "pass_rate": pass_rate,
            "attempt": len(state["attempts"]),
        }

    def _extract_code(self, messages: list[dict]) -> str:
        """Extract code block from assistant message."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Extract from markdown code block
                if "```" in content:
                    parts = content.split("```")
                    if len(parts) >= 2:
                        code = parts[1]
                        if code.startswith("python"):
                            code = code[6:]
                        return code.strip()
                return content
        return ""

    async def _run_tests(self, code: str, test_cases: list) -> dict:
        """Execute code against test cases."""
        results = {"passed": 0, "failed": [], "errors": []}

        for i, test in enumerate(test_cases):
            try:
                # Write code to temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    # Inject test input
                    full_code = f"{code}\n\nprint({test['input']})"
                    f.write(full_code)
                    f.flush()

                    # Execute
                    result = subprocess.run(
                        ["python", f.name],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout
                    )

                    output = result.stdout.strip()
                    expected = str(test["expected"]).strip()

                    if output == expected:
                        results["passed"] += 1
                    else:
                        results["failed"].append({
                            "test": i,
                            "expected": expected,
                            "got": output,
                        })

            except subprocess.TimeoutExpired:
                results["errors"].append({"test": i, "error": "Timeout"})
            except Exception as e:
                results["errors"].append({"test": i, "error": str(e)})

        return results

    def _format_test_feedback(self, results: dict) -> str:
        """Format test results as feedback."""
        lines = []

        if results["failed"]:
            lines.append("Some tests failed:")
            for f in results["failed"][:3]:  # Show first 3
                lines.append(f"  Test {f['test']}: expected {f['expected']}, got {f['got']}")

        if results["errors"]:
            lines.append("Some tests had errors:")
            for e in results["errors"][:2]:
                lines.append(f"  Test {e['test']}: {e['error']}")

        lines.append(f"\nPassed: {results['passed']}/{results['passed'] + len(results['failed']) + len(results['errors'])}")
        lines.append("Please fix your code and try again.")

        return "\n".join(lines)

    async def finalize_interaction(self, instance_id: str, **kwargs):
        if instance_id in self._instances:
            del self._instances[instance_id]
```

Config:

```yaml
# code_env/code_interaction_config.yaml
interaction:
  - name: "code_exec"
    class_name: "code_env.code_interaction.CodeExecutionInteraction"
    config:
      timeout: 5
      language: python
```

---

## Part 5: Reward Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Turn Episode                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Turn 1:                                                    │
│    Model generates → Interaction.generate_response()        │
│                      └→ turn_reward_1, feedback_1           │
│                                                             │
│  Turn 2:                                                    │
│    Model generates → Tool.execute() [if tool call]          │
│                      └→ tool_reward                         │
│                   → Interaction.generate_response()         │
│                      └→ turn_reward_2, feedback_2           │
│                                                             │
│  Turn N:                                                    │
│    Model generates → ... → should_terminate=True            │
│                                                             │
│  Final:                                                     │
│    Interaction.calculate_score() → final_score              │
│    Tool.calc_reward() → tool_final_score                    │
│                                                             │
│  Total Reward = sum(turn_rewards) + tool_rewards + final    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 6: Tips & Gotchas

### 1. Reward Shaping Matters A Lot

```python
# Bad: Sparse reward (hard to learn)
turn_reward = 1.0 if should_terminate and success else 0.0

# Better: Dense reward
turn_reward = current_score  # Reward progress

# Even better: Improvement reward
turn_reward = max(0, current_score - prev_score)
```

### 2. Feedback Quality Shapes Learning

```python
# Bad: Uninformative
feedback = "Wrong. Try again."

# Good: Actionable
feedback = f"Your answer {answer} is close but not exact. The format should be X."
```

### 3. State Management

```python
# Always clean up!
async def finalize_interaction(self, instance_id: str, **kwargs):
    if instance_id in self._instances:
        del self._instances[instance_id]
```

### 4. Handle Edge Cases

```python
async def generate_response(self, instance_id: str, messages, **kwargs):
    # Handle missing instance
    if instance_id not in self._instances:
        return True, "Error: Unknown instance", 0.0, {}

    # Handle empty response
    response = self._get_last_assistant_message(messages)
    if not response:
        return False, "No response detected. Please provide an answer.", 0.0, {}
```

### 5. Async is Required

All interaction/tool methods must be `async`:

```python
# Wrong
def generate_response(self, ...):
    pass

# Right
async def generate_response(self, ...):
    pass
```

### 6. Test Locally First

```python
# Quick test script
import asyncio
from my_env.my_interaction import MyCustomInteraction

async def test():
    interaction = MyCustomInteraction({"max_turns": 3})

    instance_id = await interaction.start_interaction(
        ground_truth="42"
    )

    messages = [
        {"role": "user", "content": "What is 6 * 7?"},
        {"role": "assistant", "content": "The answer is 40."},
    ]

    should_terminate, feedback, reward, data = await interaction.generate_response(
        instance_id, messages
    )

    print(f"Terminate: {should_terminate}")
    print(f"Feedback: {feedback}")
    print(f"Reward: {reward}")

    await interaction.finalize_interaction(instance_id)

asyncio.run(test())
```

---

## Part 7: Running Training

```bash
# 1. Install your env module
pip install -e my_env/

# 2. Run training
python -m verl.trainer.main_ppo \
    --config-path my_env/ \
    --config-name train_config \
    model.path=Qwen/Qwen2.5-3B-Instruct
```

Or with the script:

```bash
# run_training.sh
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m verl.trainer.main_ppo \
    --config-path my_env/ \
    --config-name train_config \
    model.path=Qwen/Qwen2.5-3B-Instruct \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=my_env/my_interaction_config.yaml
```

---

## Examples in verl

Check these for reference:

| Example | Files |
|---------|-------|
| GSM8K Math | `verl/interactions/gsm8k_interaction.py` |
| GSM8K Tool | `verl/tools/gsm8k_tool.py` |
| Search/RAG | `verl/tools/search_tool.py` |
| Code Sandbox | `verl/tools/sandbox_fusion_tools.py` |
| Configs | `examples/sglang_multiturn/config/` |

---

## Summary Checklist

- [ ] Create interaction class extending `BaseInteraction`
- [ ] Implement `start_interaction()` - initialize state
- [ ] Implement `generate_response()` - evaluate & feedback
- [ ] Implement `finalize_interaction()` - cleanup
- [ ] Create YAML config registering your class
- [ ] Prepare dataset with `prompt`, `data_source`, `reward_model.ground_truth`
- [ ] Update training config with `multi_turn.enable=True`
- [ ] Test locally with async test script
- [ ] Run training!
