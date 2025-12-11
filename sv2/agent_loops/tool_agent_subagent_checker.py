from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class SubagentCheckerConfig:
    enable: bool = True
    max_response_chars: int = 4000
    system_prompt: str = (
        "You are a strict checker. Predict what should happen next in a multi-turn tool-using rollout.\n"
        "Return ONLY valid JSON in a single line.\n"
        "Schema:\n"
        '{\n'
        '  "next_action": "call_tool" | "interact" | "terminate" | "continue",\n'
        '  "tool_calls": [{"name": str, "arguments": object}]  // optional\n'
        "}\n"
    )
    user_prompt_template: str = (
        "Conversation so far (messages, oldest->newest):\n"
        "{conversation_json}\n\n"
        "Assistant just said (tool calls removed):\n"
        "{assistant_content}\n\n"
        "Available tools (schemas):\n"
        "{tools_json}\n\n"
        "Predict the next action."
    )
    sampling_params: dict[str, Any] = None
    stop: Optional[list[str]] = None


def _safe_json_extract(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Best-effort extraction of a JSON object from free-form text."""
    text = (text or "").strip()
    if not text:
        return None, "empty"
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_object_found"
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate), None
    except Exception as e:
        return None, f"json_parse_error: {e}"


def _state_to_action(state: AgentState) -> str:
    if state == AgentState.PROCESSING_TOOLS:
        return "call_tool"
    if state == AgentState.INTERACTING:
        return "interact"
    if state == AgentState.TERMINATED:
        return "terminate"
    return "continue"


@register("sv2_tool_agent_checker")
class Sv2ToolAgentWithSubagentChecker(ToolAgentLoop):
    """
    ToolAgentLoop with an optional "subagent checker" call after each assistant generation.

    The checker is just another LLM call (by default using the same rollout server manager)
    that predicts the next action/tool calls. We log prediction vs actual per assistant turn.

    Enable by setting dataset `agent_name=sv2_tool_agent_checker` and loading this module
    (e.g. via `actor_rollout_ref.rollout.agent.agent_loop_config_path`).
    """

    def __init__(self, *args, checker: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        checker = checker or {}
        self.checker_cfg = SubagentCheckerConfig(
            enable=bool(checker.get("enable", True)),
            max_response_chars=int(checker.get("max_response_chars", 4000)),
            system_prompt=str(checker.get("system_prompt", SubagentCheckerConfig.system_prompt)),
            user_prompt_template=str(
                checker.get("user_prompt_template", SubagentCheckerConfig.user_prompt_template)
            ),
            sampling_params=dict(checker.get("sampling_params", {"temperature": 0.0, "top_p": 1.0})),
            stop=checker.get("stop", None),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        output = await super().run(sampling_params, **kwargs)
        # Attach any per-turn checker logs that were recorded during the run.
        checker_logs = getattr(self, "_last_checker_logs", None)
        if checker_logs is not None:
            output.extra_fields["sv2_checker"] = checker_logs
        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        next_state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

        if not self.checker_cfg.enable:
            return next_state

        # Decode assistant output and extract tool-call-stripped content.
        try:
            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
        except Exception:
            assistant_text = ""

        try:
            assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        except Exception:
            assistant_content, tool_calls = assistant_text, []

        actual = {
            "next_action": _state_to_action(next_state),
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in (tool_calls or [])],
        }

        predicted, pred_err = await self._run_checker_prediction(agent_data, assistant_content)
        record = {
            "assistant_turns": int(agent_data.assistant_turns),
            "predicted": predicted,
            "predicted_error": pred_err,
            "actual": actual,
            "assistant_text": assistant_text[: self.checker_cfg.max_response_chars],
        }

        # Store logs on the instance (AgentLoopOutput doesn't expose agent_data).
        if not hasattr(self, "_last_checker_logs") or self._last_checker_logs is None:
            self._last_checker_logs = []
        self._last_checker_logs.append(record)

        return next_state

    async def _run_checker_prediction(
        self, agent_data: AgentData, assistant_content_no_tool_calls: str
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Call the checker subagent (LLM) and parse JSON prediction."""
        try:
            conversation = list(agent_data.messages) + [{"role": "assistant", "content": assistant_content_no_tool_calls}]
            tools_json = json.dumps(self.tool_schemas or [], ensure_ascii=False)
            user_prompt = self.checker_cfg.user_prompt_template.format(
                conversation_json=json.dumps(conversation, ensure_ascii=False),
                assistant_content=assistant_content_no_tool_calls,
                tools_json=tools_json,
            )
            messages = [
                {"role": "system", "content": self.checker_cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if self.processor is not None:
                raw_prompt = await self.loop.run_in_executor(
                    None,
                    lambda: self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    ),
                )
                model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
                prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )

            sub_sampling = dict(self.checker_cfg.sampling_params or {})
            sub_sampling.setdefault("logprobs", False)
            if self.checker_cfg.stop is not None:
                sub_sampling["stop"] = self.checker_cfg.stop

            sub_out = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=prompt_ids,
                sampling_params=sub_sampling,
                image_data=None,
            )
            text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(sub_out.token_ids, skip_special_tokens=True)
            )
            parsed, err = _safe_json_extract(text)
            return parsed, err
        except Exception as e:
            return None, f"checker_exception: {e}"

