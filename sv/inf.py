"""
Inference utilities: tokenize prompts, build rollout manager, and run multi-turn/tool rollouts.
"""

from __future__ import annotations

from typing import List, Tuple

from omegaconf import OmegaConf
import torch

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils import hf_processor, hf_tokenizer


def build_tokenizer_processor(model_cfg: dict) -> Tuple[object, object]:
    tokenizer = hf_tokenizer(model_cfg["path"], trust_remote_code=model_cfg.get("trust_remote_code", False))
    processor = hf_processor(model_cfg["path"], trust_remote_code=model_cfg.get("trust_remote_code", False), use_fast=True)
    return tokenizer, processor


def build_agent_loop_manager(cfg: dict) -> AgentLoopManager:
    # Requires Ray; assumes single node/single GPU as per config defaults
    return AgentLoopManager(config=OmegaConf.create(cfg), worker_group=None, rm_resource_pool=None)


def tokenize_prompts(prompts: List[str], tokenizer, max_length: int) -> dict:
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
    }


def run_rollout(cfg: dict, agent_loop_mgr: AgentLoopManager, batch_dict: dict, tokenizer=None):
    batch = DataProto.from_single_dict(batch_dict)

    # Prepare generation batch
    gen_batch = batch.pop(batch_keys=[k for k in ["input_ids", "attention_mask", "position_ids"] if k in batch.batch])
    gen_batch.meta_info.update(
        {
            "do_sample": True,
            "global_steps": 0,
        }
    )

    # Multi-turn rollout
    gen_out = agent_loop_mgr.generate_sequences(gen_batch)
    batch = batch.union(gen_out)

    # Ensure response mask available
    if "response_mask" not in batch.batch:
        batch.batch["response_mask"] = compute_response_mask(batch)

    # Optionally attach decoded responses for debugging
    if tokenizer is not None and "responses" in batch.batch:
        decoded = [tokenizer.decode(resp, skip_special_tokens=True) for resp in batch.batch["responses"]]
        batch.non_tensor_batch["decoded_responses"] = decoded

    return batch
