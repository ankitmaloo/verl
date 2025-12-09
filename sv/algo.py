"""
Orchestration: prompts -> rollout -> reward -> advantages -> ready for custom loss.
Use this as the entrypoint to swap datasets/tasks by changing prompts and config.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Callable, List

import numpy as np
import ray
import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from sv import inf, train
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_advantage
from verl.trainer.ppo.ray_trainer import compute_response_mask


def load_config(config_path: str | pathlib.Path):
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def default_reward_fn(batch) -> tuple[torch.Tensor, dict]:
    """
    Placeholder reward: zeros; replace with your scorer.
    """
    responses = batch.batch["responses"]
    rewards = torch.zeros_like(responses, dtype=torch.float32)
    return rewards, {}


def run(
    prompts: List[str],
    config_path: str = "sv/gsm8k_multi_config.yaml",
    reward_fn: Callable = default_reward_fn,
):
    cfg = load_config(config_path)

    # Init Ray once
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Build tokenizer/processor and agent loop
    tokenizer, processor = inf.build_tokenizer_processor(cfg["actor_rollout_ref"]["model"])
    agent_loop_mgr = inf.build_agent_loop_manager(cfg)

    # Tokenize prompts
    batch_dict = inf.tokenize_prompts(
        prompts,
        tokenizer=tokenizer,
        max_length=cfg["data"]["max_prompt_length"],
    )

    # Rollout (multi-turn/tools)
    batch = inf.run_rollout(cfg, agent_loop_mgr, batch_dict, tokenizer=tokenizer)

    # Reward
    rewards, reward_extra = reward_fn(batch)
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

    # Custom loss/weight update hook
    train.update_policy_stub(batch)

    # Decode a couple of responses for inspection
    decoded = [tokenizer.decode(resp, skip_special_tokens=True) for resp in batch.batch["responses"]]
    batch.non_tensor_batch["decoded_responses"] = np.array(decoded, dtype=object)

    return batch


def _load_prompts_from_args(args) -> List[str]:
    if args.prompts_json:
        prompts = json.loads(args.prompts_json)
        if not isinstance(prompts, list):
            raise ValueError("prompts_json must be a JSON array of strings")
        return prompts
    if args.hf_dataset:
        ds = load_dataset(args.hf_dataset, args.hf_subset or None, split=args.hf_split)
        field = args.hf_prompt_field
        if field not in ds.column_names:
            raise ValueError(f"Field '{field}' not in dataset columns {ds.column_names}")
        return list(ds[field])
    return ["You are a math tutor. Solve: If Johnny has 3 apples and buys 4 more, how many apples does he have?"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sv/gsm8k_multi_config.yaml")
    parser.add_argument("--prompts-json", type=str, help="JSON array of prompts")
    parser.add_argument("--hf-dataset", type=str, help="HF dataset name (e.g., openai/gsm8k)")
    parser.add_argument("--hf-subset", type=str, default=None)
    parser.add_argument("--hf-split", type=str, default="test")
    parser.add_argument("--hf-prompt-field", type=str, default="question")
    args = parser.parse_args()

    prompts = _load_prompts_from_args(args)
    output_batch = run(prompts, config_path=args.config)

    responses = output_batch.batch.get("responses")
    advantages = output_batch.batch.get("advantages")
    print("responses shape:", responses.shape if responses is not None else None)
    print("advantages shape:", advantages.shape if advantages is not None else None)
    decoded = output_batch.non_tensor_batch.get("decoded_responses")
    if decoded is not None:
        print("sample decoded responses:", decoded[:2])
