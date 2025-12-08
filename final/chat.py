#!/usr/bin/env python3
"""Simple interactive chat with the inference engine.

Usage:
    python chat.py        # Basic chat
    python chat.py tool   # Chat with add tool enabled
"""
import sys
import textwrap


def basic_chat():
    """Basic interactive chat."""
    from inf import InferenceEngine
    engine = InferenceEngine("config.yaml")
    while (q := input("\nYou: ").strip()):
        print("\nAI:", textwrap.fill(engine.generate(q).completions[0], width=80))


# =============================================================================
# Tool Calling Example
# =============================================================================

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers together",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
}

def tool_chat():
    """Interactive chat with add tool enabled."""
    from inf import InferenceEngine
    engine = InferenceEngine("config.yaml")
    tools = [ADD_TOOL]
    print("\nTool chat - try: 'What is 123 + 456?'\n")
    while (q := input("You: ").strip()):
        output = engine.generate(q, tools=tools, temperature=0)
        if output.tool_calls and output.tool_calls[0]:
            tc = output.tool_calls[0][0]
            print(f"[Tool: {tc.name}({tc.arguments})]")
            result = add(**tc.arguments)
            print(f"[Result: {result}]")
            output = engine.generate_with_tool_result(
                messages=output.messages[0], tool_call=tc,
                tool_result=str(result), tools=tools, temperature=0,
            )
        print(f"\nAI: {textwrap.fill(output.completions[0], width=80)}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tool":
        tool_chat()
    else:
        basic_chat()
