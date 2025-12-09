"""
Local Python code execution tool for tool-calling rollouts.
Runs code in a subprocess with a timeout and returns stdout/stderr to the LLM.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse


class LocalPythonTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.timeout = int(config.get("timeout", 10))
        self.max_output_chars = int(config.get("max_output_chars", 4000))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        output = await asyncio.to_thread(self._run_code, code)
        return ToolResponse(text=output), 0.0, {}

    def _run_code(self, code: str) -> str:
        wrapped = textwrap.dedent(code).strip() + "\n"
        with tempfile.NamedTemporaryFile("w", delete=True, suffix=".py") as tmp:
            tmp.write(wrapped)
            tmp.flush()
            try:
                result = subprocess.run(
                    [sys.executable, tmp.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout,
                    text=True,
                )
                output = (result.stdout or "") + (result.stderr or "")
            except subprocess.TimeoutExpired:
                output = f"Execution timed out after {self.timeout} seconds."
            except Exception as exc:  # pragma: no cover - defensive
                output = f"Execution failed: {exc}"

        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "...(truncated)"
        return output
