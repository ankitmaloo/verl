#!/usr/bin/env python3
"""
Rollout-only driver (no optimizer step) using the agent-loop stack.

What this does:
- Loads a training-style config (defaults toward examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml)
- Starts Ray, rollout server (vLLM/sglang per config), AgentLoopManager
- Runs batched multi-turn rollouts (default GSM8K, batch size 32)
- Keeps thinking tokens for loss; masks them from reward; final reward only
- Leaves placeholders where training/optimizer would normally run (do not remove)

Assumptions:
- Model: from config (default Qwen3-4B)
- Tool config: from config
- Interaction config: from config
- Single node, single-GPU-friendly; still uses Ray to mirror training stack

Notes on outputs:
- rollout_out.batch contains token-level tensors (including special tokens); decoding/text trimming is left to downstream code.
"""

import argparse
import json
from pathlib import Path

import sys
import importlib
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf

# Ensure repo root is on sys.path for imports like verl_research.*
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from verl.utils.debug import marked_timer
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl_research.tools.training_step import training_step


def load_hydra_config(cfg_path: Path):
    """Load a Hydra-style YAML and resolve defaults (ppo_trainer, etc.)."""
    cfg_path = cfg_path.resolve()
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as e:
        raise ImportError("hydra-core is required for rollout_only config resolution. pip install hydra-core") from e

    # Add trainer config dir so defaults like `ppo_trainer` can be found when running from repo root
    repo_root = Path(__file__).resolve().parents[2]
    trainer_cfg_dir = (repo_root / "verl" / "trainer" / "config").resolve()
    overrides = []
    if trainer_cfg_dir.exists():
        overrides.append(f"+hydra.searchpath=[file://{trainer_cfg_dir}]")

    with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent)):
        cfg = compose(config_name=cfg_path.stem, overrides=overrides)
    return cfg


def override_for_rollout_only(cfg, *, batch_size: int):
    """Adjust config for rollout-only run (no optimizer)."""
    cfg = OmegaConf.merge(cfg, {})
    # Ensure multi_turn enable if using multiturn example config
    if "actor_rollout_ref" in cfg and "rollout" in cfg.actor_rollout_ref:
        cfg.actor_rollout_ref.rollout.multi_turn.enable = True
        cfg.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = cfg.actor_rollout_ref.rollout.multi_turn.get(
            "max_assistant_turns", 8
        )
        cfg.actor_rollout_ref.rollout.calculate_log_probs = True
        cfg.data.train_batch_size = batch_size
    # Training placeholders stay; we simply won't call optimizer
    return cfg


def init_ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def build_agent_loop_manager(cfg: DictConfig) -> AgentLoopManager:
    from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager

    # Reuse helper that builds Ray worker group + AgentLoopManager (hybrid, colocated)
    agent_loop_manager = init_agent_loop_manager(cfg)
    return agent_loop_manager


def dataloader(dataset, batch_size: int):
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def collate_batch(batch, tokenizer):
    """Convert list of dicts into DataProto for rollout."""
    prompts = [ex["raw_prompt"] for ex in batch]
    # pad/truncate via tokenizer chat template
    # RLHFSequenceDataset already tokenized; use its outputs
    # here we just pass through
    non_tensor_batch = {"raw_prompt": np.array(prompts, dtype=object)}
    data = DataProto(batch=None, non_tensor_batch=non_tensor_batch, meta_info={"validate": False})
    return data


def load_task_module(module_path: str):
    """
    Dynamically load a task module that provides:
      - build_dataset(tokenizer, max_prompt_len, max_response_len, split, limit)
      - optional reward/interaction helpers (not wired automatically)
    Default: verl_research.tools.gsm8k_v
    """
    return importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser(description="Rollout-only driver with agent loop stack (batched).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("verl-research/tools/rollout_config_gsm8k.yaml"),
        help="Trainer-style YAML config.",
    )
    parser.add_argument("--split", default="train[:64]", help="Dataset split, e.g., 'train[:64]'")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit total examples.")
    parser.add_argument("--output", type=Path, default=Path("rollout_only_results.jsonl"))
    parser.add_argument(
        "--task-module",
        default="verl_research.tools.gsm8k_v",
        help="Module providing build_dataset(tokenizer, max_prompt_len, max_response_len, split, limit).",
    )
    args = parser.parse_args()

    init_ray()

    raw_cfg = load_hydra_config(args.config)
    cfg = override_for_rollout_only(raw_cfg, batch_size=args.batch_size)
    cfg_dc: DictConfig = cfg

    # Build AgentLoopManager (spins rollout server)
    with marked_timer("init_agent_loop_manager"):
        agent_loop_manager = build_agent_loop_manager(cfg_dc)

    # Tokenizer for dataset
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(cfg_dc.actor_rollout_ref.model.path, trust_remote_code=True)

    task_mod = load_task_module(args.task_module)
    if not hasattr(task_mod, "build_dataset"):
        raise AttributeError(f"{args.task_module} must define build_dataset(...)")
    dataset = task_mod.build_dataset(
        tokenizer,
        max_prompt_len=cfg_dc.data.max_prompt_length,
        max_response_len=cfg_dc.data.max_response_length,
        split=args.split,
        limit=args.limit,
    )

    out_f = args.output.open("w")

    # Rollout loop (batched)
    for batch in dataloader(dataset, args.batch_size):
        data = collate_batch(batch, tokenizer)
        # pad to divisor (mirrors trainer behavior)
        data, pad_size = pad_dataproto_to_divisor(data, size_divisor=cfg_dc.trainer.n_gpus_per_node)

        # === Rollout ===
        with marked_timer("generate_sequences"):
            rollout_out: DataProto = agent_loop_manager.generate_sequences(data)
        rollout_out = unpad_dataproto(rollout_out, pad_size)

        # Training step placeholder (no optimizer updates)
        train_metrics = training_step(rollout_out, cfg_dc)

        # Collect final rewards if present
        rewards = rollout_out.batch.get("rm_scores", None)

        # Decode outputs for inspection/logging; keep special tokens
        decoded = []
        if "responses" in rollout_out.batch:
            responses = rollout_out.batch["responses"]
            for resp_ids in responses:
                decoded.append(tokenizer.decode(resp_ids.tolist(), skip_special_tokens=False))

        meta = rollout_out.meta_info
        record = {
            "batch_size": len(batch),
            "rm_scores": rewards.tolist() if rewards is not None else None,
            "decoded_responses": decoded,
            "meta": meta,
            "train_metrics": train_metrics,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

    out_f.close()
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
