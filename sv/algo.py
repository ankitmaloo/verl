"""
Algorithm + orchestration in one place:
- fetch prompts from a task file
- run rollouts (multi-turn capable per config)
- hand responses back to the task for more turns
- compute reward/advantages and hand off to train.py
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from sv import inf, train
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_advantage
from verl.trainer.ppo.ray_trainer import compute_response_mask

DEFAULT_TASK = "sv.task_gsm8k"


def load_config(config_path: str | pathlib.Path):
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def default_reward_fn(batch, state=None) -> tuple[torch.Tensor, dict]:
    """Placeholder reward: zeros; replace with your scorer."""
    responses = batch.batch["responses"]
    rewards = torch.zeros_like(responses, dtype=torch.float32)
    return rewards, {}


def _load_task(task_path: str):
    """
    Task is a module path like `sv.task_example`.
    The module may expose:
    - init_state(config) -> state
    - get_initial_prompts(config, state) -> List[str]
    - process_responses(batch, config, state) -> Dict with next_prompts, done, reward_fn
    - build_reward_fn(tokenizer, config, state) -> callable
    """
    return importlib.import_module(task_path)


def _get_max_turns(cfg: Dict[str, Any]) -> int:
    multi_turn = cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("multi_turn", {})
    configured = int(multi_turn.get("max_assistant_turns", 1))
    return min(10, configured)


def run(
    prompts: Optional[List[str]] = None,
    config_path: str = "sv/gsm8k_multi_config.yaml",
    task_module: str = DEFAULT_TASK,
    reward_fn: Optional[Callable] = None,
):
    cfg = load_config(config_path)

    # Init Ray once
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    task = _load_task(task_module)
    state = task.init_state(cfg) if hasattr(task, "init_state") else None

    # Build tokenizer/processor and agent loop
    tokenizer, processor = inf.build_tokenizer_processor(cfg["actor_rollout_ref"]["model"])
    agent_loop_mgr = inf.build_agent_loop_manager(cfg)

    # Task-provided prompts
    prompts = prompts or task.get_initial_prompts(cfg, state)

    # Reward hook
    reward_fn = reward_fn or (
        task.build_reward_fn(tokenizer=tokenizer, config=cfg, state=state)
        if hasattr(task, "build_reward_fn")
        else default_reward_fn
    )

    max_turns = _get_max_turns(cfg)
    history_batches = []
    batch = None

    for turn in range(max_turns):
        # Tokenize prompts for this turn
        batch_dict = inf.tokenize_prompts(
            prompts,
            tokenizer=tokenizer,
            max_length=cfg["data"]["max_prompt_length"],
        )

        # Rollout (multi-turn/tools)
        batch = inf.run_rollout(cfg, agent_loop_mgr, batch_dict, tokenizer=tokenizer)
        history_batches.append(batch)

        # Task callback for next prompts
        if hasattr(task, "process_responses"):
            step_out = task.process_responses(batch=batch, config=cfg, state=state) or {}
            next_prompts = step_out.get("next_prompts")
            done = bool(step_out.get("done", False))
            # Allow task to swap reward_fn mid-run
            reward_fn = step_out.get("reward_fn", reward_fn)
        else:
            next_prompts, done = None, True

        if done or not next_prompts:
            break
        prompts = next_prompts

    if batch is None:
        raise RuntimeError("No rollout executed; check prompts/task wiring.")

    # Reward
    rewards, reward_extra = reward_fn(batch, state=state)
    batch.batch["token_level_scores"] = rewards
    if reward_extra:
        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra.items()})

    # Ensure response mask exists
    if "response_mask" not in batch.batch:
        batch.batch["response_mask"] = compute_response_mask(batch)

    # Advantages (GRPO outcome style)
    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        gamma=cfg["algorithm"]["gamma"],
        lam=cfg["algorithm"]["lam"],
        num_repeat=cfg["actor_rollout_ref"]["rollout"].get("n", 1),
        norm_adv_by_std_in_grpo=cfg["algorithm"].get("norm_adv_by_std_in_grpo", True),
        config=OmegaConf.create(cfg["algorithm"]),
    )

    # Custom loss/weight update hook (no-op for eval-only flows)
    train.update_policy_stub(batch)

    # Decode a couple of responses for inspection
    decoded = [tokenizer.decode(resp, skip_special_tokens=True) for resp in batch.batch["responses"]]
    batch.non_tensor_batch["decoded_responses"] = np.array(decoded, dtype=object)
    batch.non_tensor_batch["history_len"] = len(history_batches)

    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sv/gsm8k_multi_config.yaml")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task module path, e.g., sv.task_gsm8k")
    parser.add_argument(
        "--prompts-json",
        type=str,
        help="Optional JSON array of prompts to override task initial prompts.",
    )
    args = parser.parse_args()

    prompts_override = None
    if args.prompts_json:
        import json

        prompts_override = json.loads(args.prompts_json)
        if not isinstance(prompts_override, list):
            raise ValueError("prompts-json must decode to a list of strings")

    output_batch = run(prompts_override, config_path=args.config, task_module=args.task)

    responses = output_batch.batch.get("responses")
    advantages = output_batch.batch.get("advantages")
    print("responses shape:", responses.shape if responses is not None else None)
    print("advantages shape:", advantages.shape if advantages is not None else None)
    decoded = output_batch.non_tensor_batch.get("decoded_responses")
    if decoded is not None:
        print("sample decoded responses:", decoded[:2])
