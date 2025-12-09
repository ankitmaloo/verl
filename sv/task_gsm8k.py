"""
GSM8K task with tool-calling support.
- Loads GSM8K test split questions as prompts.
- Lets the model decide when to call the code_interpreter tool.
- Computes rewards via exact answer match (normalized).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset


@dataclass
class Gsm8kExample:
    question: str
    answer: str


def _normalize_scalar(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    # Extract the value after "####" if present
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        text = match.group(1).strip()

    # Fallback to last number in the string if no explicit marker
    if not re.search(r"[-+]?\d", text):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if nums:
            text = nums[-1]

    return text.strip()


def _as_number(val: str):
    try:
        if "/" in val:
            return Fraction(val)
        return Fraction(val)
    except Exception:
        return val


def _eq_answer(pred: str, gold: str) -> bool:
    pred_norm = _normalize_scalar(pred)
    gold_norm = _normalize_scalar(gold)
    p_num, g_num = _as_number(pred_norm), _as_number(gold_norm)
    if isinstance(p_num, Fraction) and isinstance(g_num, Fraction):
        return p_num == g_num
    return pred_norm.strip().lower() == gold_norm.strip().lower()


def init_state(config: Dict[str, Any]) -> Dict[str, Any]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    max_samples = config.get("task", {}).get("max_samples")
    if max_samples is not None:
        ds = ds.select(range(int(max_samples)))

    examples = [Gsm8kExample(q["question"], q["answer"]) for q in ds]
    return {"turn": 0, "examples": examples, "logs": []}


def _build_prompt(q: str) -> str:
    return (
        "You are a meticulous math tutor. Solve the problem step by step. "
        "Use the code_interpreter tool if calculations help. "
        "Return the final numeric answer formatted as '#### <answer>'.\n\n"
        f"Question:\n{q}"
    )


def get_initial_prompts(config: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    return [_build_prompt(ex.question) for ex in state["examples"]]


def process_responses(batch, config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    decoded = batch.non_tensor_batch.get("decoded_responses")
    entry: Dict[str, Any] = {"turn": state["turn"]}
    if decoded is not None:
        entry["decoded"] = decoded

    tool_info = batch.non_tensor_batch.get("tool_extra_fields")
    if tool_info is not None:
        entry["tool_extra_fields"] = tool_info

    state["logs"].append(entry)

    state["turn"] += 1
    max_turns = min(10, config.get("actor_rollout_ref", {}).get("rollout", {}).get("multi_turn", {}).get("max_assistant_turns", 10))
    has_final = False
    if decoded is not None:
        has_final = any("####" in resp for resp in decoded)

    done = has_final or state["turn"] >= max_turns
    batch.non_tensor_batch["gsm8k_logs"] = np.array(state["logs"], dtype=object)
    return {"done": done}


def build_reward_fn(tokenizer=None, config=None, state=None):
    answers = [ex.answer for ex in state["examples"]]

    def reward_fn(batch, state=None) -> Tuple[torch.Tensor, dict]:
        responses = batch.batch["responses"]
        decoded = batch.non_tensor_batch.get("decoded_responses")
        if decoded is None:
            decoded = [tokenizer.decode(r, skip_special_tokens=True) for r in responses]

        token_rewards = torch.zeros_like(responses, dtype=torch.float32)
        eval_logs = []
        for i, (pred, gold) in enumerate(zip(decoded, answers)):
            correct = 1.0 if _eq_answer(pred, gold) else 0.0
            token_rewards[i, :] = correct
            eval_logs.append({"pred": pred, "gold": gold, "correct": bool(correct)})

        batch.non_tensor_batch["reward_logs"] = np.array(eval_logs, dtype=object)
        return token_rewards, {}

    return reward_fn
