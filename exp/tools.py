import json
import re

TOOLS = {
    "calculator": lambda operation, a, b: {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "error: division by zero"
    }.get(operation, f"unknown op: {operation}"),

    "search": lambda query: f"[search results for: {query}]",

    "get_weather": lambda location: f"Weather in {location}: 72°F, sunny",
}


def parse_tool_call(response):
    # Qwen/OpenAI format: {"function_call": {"name": "...", "arguments": "..."}}
    if isinstance(response, dict):
        if "function_call" in response:
            fc = response["function_call"]
            args = fc.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            return fc["name"], args
        if "tool_calls" in response:
            tc = response["tool_calls"][0]["function"]
            args = tc.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            return tc["name"], args

    # String response - parse various formats
    if isinstance(response, str):
        # Hermes format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return data.get("name"), data.get("arguments", {})

        # Qwen format: <|im_start|>assistant\n{"function_call": ...}
        match = re.search(r'\{"function_call":\s*\{[^}]+\}\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            fc = data["function_call"]
            args = fc.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            return fc["name"], args

        # Direct JSON: {"name": "...", "arguments": {...}}
        match = re.search(r'\{"name":\s*"(\w+)"[^}]*"arguments":\s*(\{[^}]*\})', response, re.DOTALL)
        if match:
            return match.group(1), json.loads(match.group(2))

    return None, None


def execute_tool(name, args):
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'"}
    try:
        if isinstance(args, str):
            args = json.loads(args)
        result = TOOLS[name](**args)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def process_response_with_tools(response):
    name, args = parse_tool_call(response)
    if name:
        result = execute_tool(name, args)
        return True, name, args, json.dumps(result)
    return False, None, None, None


def format_tool_result(name, result):
    return {"role": "function", "name": name, "content": result}
