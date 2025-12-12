import ast
import asyncio
import io
import sys
from typing import Any, Tuple

from verl.tools.base_tool import BaseTool, ToolResponse, register_tool
from verl.tools.schemas import ToolSchema

@register_tool("python_exec")
class PythonExecTool(BaseTool):
    """
    A tool that executes Python code and returns the stdout.
    WARNING: This executes code with limited isolation. Do not use with untrusted code/users.
    """
    
    def __init__(self, tool_schema: ToolSchema):
        super().__init__(tool_schema)
        # Sandbox / Safety limits could be added here
        self.max_output_len = 1000

    async def create(self, create_kwargs: dict[str, Any]) -> Tuple[str, dict[str, Any]]:
        # No setup needed per instance
        return "local_session", {}

    async def execute(self, instance_id: str, tool_args: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict[str, Any]]:
        code = tool_args.get("code", "")
        if not code:
            return ToolResponse(text="Error: No code provided."), 0.0, {}

        loop = asyncio.get_running_loop()

        def _exec_sync():
            # Capture stdout
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output

            try:
                # We use ast.parse to validate it's valid python (optional)
                ast.parse(code)
                
                # Execute in a restricted namespace
                # Note: This is NOT a secure sandbox.
                local_scope = {}
                exec(code, {"__builtins__": __builtins__}, local_scope)
                
                output = redirected_output.getvalue()
                if not output and "result" in local_scope:
                    # If no stdout but 'result' variable exists, return that
                    output = str(local_scope["result"])
                elif not output:
                     output = "[Code executed successfully with no output]"

            except Exception as e:
                output = f"Execution Error: {str(e)}"
            finally:
                sys.stdout = old_stdout
            
            return output

        # Run in executor to avoid blocking the async loop
        output = await loop.run_in_executor(None, _exec_sync)

        # Truncate if too long
        if len(output) > self.max_output_len:
            output = output[:self.max_output_len] + "...(truncated)"

        return ToolResponse(text=output), 0.0, {}

    async def release(self, instance_id: str):
        pass
