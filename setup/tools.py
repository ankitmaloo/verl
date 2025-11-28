"""
Tool/Action Management for Multi-Turn RL.

Handles:
- Tool definition and registry
- Tool calling format (JSON, ReAct, etc.)
- Action parsing from model outputs
- Tool execution and result handling
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """Definition of a tool/action the model can call."""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
        }


class Tool(ABC):
    """Base class for tools/actions."""
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool.
        
        Args:
            **kwargs: Arguments matching the tool parameters
            
        Returns:
            String result of tool execution
        """
        pass


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.definitions: Dict[str, ToolDefinition] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        name = tool.definition.name
        self.tools[name] = tool
        self.definitions[name] = tool.definition
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self.definitions.get(name)
    
    def list_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return list(self.definitions.values())
    
    def get_tools_prompt(self) -> str:
        """Get formatted tools description for model context."""
        if not self.tools:
            return ""
        
        prompt = "Available tools:\n"
        for definition in self.list_definitions():
            prompt += f"\n{definition.name}: {definition.description}\n"
            prompt += "Parameters:\n"
            for param in definition.parameters:
                required_str = "required" if param.required else "optional"
                prompt += f"  - {param.name} ({param.type}, {required_str}): {param.description}\n"
        
        return prompt
    
    def execute_tool(self, tool_name: str, **kwargs) -> Tuple[bool, str]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments
            
        Returns:
            (success: bool, result: str)
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return False, f"Tool '{tool_name}' not found"
        
        try:
            result = tool.execute(**kwargs)
            return True, result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return False, f"Tool execution error: {str(e)}"


# ============================================================================
# ACTION PARSING
# ============================================================================

@dataclass
class ParsedAction:
    """Parsed tool call from model output."""
    tool_name: str
    parameters: Dict[str, Any]
    raw_text: str
    success: bool = True
    error: Optional[str] = None


class ActionParser(ABC):
    """Base class for parsing actions from model output."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
    
    @abstractmethod
    def parse(self, text: str) -> Optional[ParsedAction]:
        """
        Parse action from text.
        
        Args:
            text: Model output text
            
        Returns:
            ParsedAction or None if parsing failed
        """
        pass


class JSONActionParser(ActionParser):
    """Parse tool calls in JSON format."""
    
    def parse(self, text: str) -> Optional[ParsedAction]:
        """
        Parse JSON tool calls.
        
        Expects format: {"tool": "tool_name", "parameters": {...}}
        or:            [{"tool": "...", "parameters": {...}}]
        """
        text = text.strip()
        
        # Try to extract JSON from text (in case model outputs extra text)
        json_match = re.search(r'\{.*?\}|\[.*?\]', text, re.DOTALL)
        if not json_match:
            return None
        
        try:
            json_text = json_match.group(0)
            data = json.loads(json_text)
            
            # Handle array (multiple tools)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            
            # Extract tool and parameters
            tool_name = data.get("tool") or data.get("name")
            parameters = data.get("parameters") or data.get("args", {})
            
            if not tool_name:
                return ParsedAction(
                    tool_name="",
                    parameters={},
                    raw_text=text,
                    success=False,
                    error="No 'tool' or 'name' field in JSON",
                )
            
            return ParsedAction(
                tool_name=tool_name,
                parameters=parameters,
                raw_text=text,
                success=True,
            )
        
        except json.JSONDecodeError as e:
            return ParsedAction(
                tool_name="",
                parameters={},
                raw_text=text,
                success=False,
                error=f"JSON parse error: {str(e)}",
            )


class ReActActionParser(ActionParser):
    """Parse actions in ReAct (Reason + Act) format."""
    
    def parse(self, text: str) -> Optional[ParsedAction]:
        """
        Parse ReAct format.
        
        Expects: "Thought: ... Action: tool_name Action Input: {...}"
        """
        text = text.strip()
        
        # Extract Action and Action Input
        action_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
        action_input_match = re.search(r'Action Input:\s*({.*?}|\w+(?:\s+\w+)*)', text, re.IGNORECASE | re.DOTALL)
        
        if not action_match:
            return None
        
        tool_name = action_match.group(1)
        
        # Try to parse parameters as JSON
        parameters = {}
        if action_input_match:
            input_text = action_input_match.group(1)
            try:
                parameters = json.loads(input_text)
            except json.JSONDecodeError:
                # Try to parse as simple key=value pairs
                try:
                    # Simple parsing of "key1=value1 key2=value2"
                    pairs = re.findall(r'(\w+)=(["\']?)([^"\']*)\2', input_text)
                    parameters = {k: v for k, _, v in pairs}
                except:
                    parameters = {"input": input_text}
        
        return ParsedAction(
            tool_name=tool_name,
            parameters=parameters,
            raw_text=text,
            success=True,
        )


class FunctionCallingParser(ActionParser):
    """Parse OpenAI function calling format."""
    
    def parse(self, text: str) -> Optional[ParsedAction]:
        """
        Parse function calling format.
        
        Expects: <function_calls>{"tool_name": "...", "parameters": {...}}</function_calls>
        """
        text = text.strip()
        
        # Look for function_calls tags
        match = re.search(
            r'<function_calls>(.*?)</function_calls>',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            return None
        
        json_text = match.group(1).strip()
        
        try:
            data = json.loads(json_text)
            
            # Support multiple formats
            tool_name = data.get("tool_name") or data.get("name") or data.get("function")
            parameters = data.get("parameters") or data.get("args", {})
            
            if not tool_name:
                return ParsedAction(
                    tool_name="",
                    parameters={},
                    raw_text=text,
                    success=False,
                    error="No tool name in function_calls",
                )
            
            return ParsedAction(
                tool_name=tool_name,
                parameters=parameters,
                raw_text=text,
                success=True,
            )
        
        except json.JSONDecodeError as e:
            return ParsedAction(
                tool_name="",
                parameters={},
                raw_text=text,
                success=False,
                error=f"JSON parse error in function_calls: {str(e)}",
            )


# ============================================================================
# MULTI-TURN INTERACTION MANAGER
# ============================================================================

@dataclass
class ToolCall:
    """Record of a tool call in multi-turn."""
    tool_name: str
    parameters: Dict[str, Any]
    result: str
    success: bool


@dataclass
class Turn:
    """Single turn in multi-turn interaction."""
    turn_number: int
    model_output: str
    parsed_action: Optional[ParsedAction]
    tool_call: Optional[ToolCall]
    observation: str


class MultiTurnManager:
    """Manages multi-turn interaction with tools and environment."""
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        action_parser: ActionParser,
        environment: Optional[Any] = None,
        max_turns: int = 5,
    ):
        """
        Initialize multi-turn manager.
        
        Args:
            tool_registry: Registry of available tools
            action_parser: Parser for extracting actions from model output
            environment: Optional environment to execute actions in
            max_turns: Maximum turns per episode
        """
        self.tool_registry = tool_registry
        self.action_parser = action_parser
        self.environment = environment
        self.max_turns = max_turns
        self.turns: List[Turn] = []
        self.conversation_history: List[Dict[str, str]] = []
    
    def reset(self):
        """Reset for new episode."""
        self.turns = []
        self.conversation_history = []
    
    def step(
        self,
        model_output: str,
        environment_obs: Optional[str] = None,
    ) -> Turn:
        """
        Process one turn.
        
        Args:
            model_output: Raw output from model
            environment_obs: Current environment observation
            
        Returns:
            Turn object with parsed action and results
        """
        turn_num = len(self.turns) + 1
        
        # Parse action from model output
        parsed_action = self.action_parser.parse(model_output)
        
        # Execute tool if parsing succeeded
        tool_call = None
        if parsed_action and parsed_action.success:
            success, result = self.tool_registry.execute_tool(
                parsed_action.tool_name,
                **parsed_action.parameters
            )
            
            tool_call = ToolCall(
                tool_name=parsed_action.tool_name,
                parameters=parsed_action.parameters,
                result=result,
                success=success,
            )
        else:
            # Parsing failed or no tool
            result = parsed_action.error if parsed_action else "Failed to parse action"
        
        # Get observation (from environment or provided)
        if environment_obs is not None:
            observation = environment_obs
        elif self.environment is not None:
            # Try to get obs from environment
            try:
                observation = self.environment.get_state_description()
            except:
                observation = "No observation available"
        else:
            observation = tool_call.result if tool_call else result
        
        # Create turn record
        turn = Turn(
            turn_number=turn_num,
            model_output=model_output,
            parsed_action=parsed_action,
            tool_call=tool_call,
            observation=observation,
        )
        
        self.turns.append(turn)
        
        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": model_output})
        self.conversation_history.append({"role": "user", "content": observation})
        
        return turn
    
    def get_system_prompt(self) -> str:
        """Get system prompt with tool descriptions."""
        tools_text = self.tool_registry.get_tools_prompt()
        
        base_prompt = """You are an AI agent capable of using tools to accomplish tasks.

When you want to use a tool, respond ONLY with valid JSON in this exact format:
{"tool": "tool_name", "parameters": {"param1": value1, "param2": value2}}

Execute one tool per response. Always provide necessary parameters.
"""
        
        return base_prompt + "\n" + tools_text
    
    def should_continue(self) -> bool:
        """Check if episode should continue."""
        if len(self.turns) >= self.max_turns:
            return False
        
        # Could add other termination conditions
        return True
    
    def get_next_prompt(self) -> str:
        """Get formatted prompt for next model call."""
        if not self.turns:
            return ""
        
        last_turn = self.turns[-1]
        return last_turn.observation
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the episode."""
        successful_tools = sum(
            1 for t in self.turns
            if t.tool_call and t.tool_call.success
        )
        
        tool_names = [
            t.tool_call.tool_name for t in self.turns
            if t.tool_call
        ]
        
        return {
            "num_turns": len(self.turns),
            "num_tool_calls": sum(1 for t in self.turns if t.tool_call),
            "successful_tool_calls": successful_tools,
            "tools_used": list(set(tool_names)),
            "turns": [
                {
                    "num": t.turn_number,
                    "tool": t.tool_call.tool_name if t.tool_call else None,
                    "success": t.tool_call.success if t.tool_call else False,
                    "observation_length": len(t.observation),
                }
                for t in self.turns
            ],
        }


# ============================================================================
# EXAMPLE TOOLS
# ============================================================================

class EnvironmentStepTool(Tool):
    """Tool for executing environment step."""
    
    def __init__(self, environment):
        self.environment = environment
        definition = ToolDefinition(
            name="step",
            description="Execute an action in the environment",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="The action to execute",
                    required=True,
                ),
            ],
        )
        super().__init__(definition)
    
    def execute(self, action: str, **kwargs) -> str:
        """Execute action in environment."""
        obs, reward, done, info = self.environment.step(action)
        return f"Observation: {obs}\nReward: {reward}\nDone: {done}"


class QueryTool(Tool):
    """Example tool for querying information."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="query",
            description="Query for information",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The query string",
                    required=True,
                ),
            ],
        )
        super().__init__(definition)
    
    def execute(self, query: str, **kwargs) -> str:
        """Execute query."""
        # TODO: Implement your query logic
        return f"Result for query: {query}"


class CalculateTool(Tool):
    """Example tool for calculations."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="calculate",
            description="Perform a calculation",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate",
                    required=True,
                ),
            ],
        )
        super().__init__(definition)
    
    def execute(self, expression: str, **kwargs) -> str:
        """Execute calculation."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
