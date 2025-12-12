"""
Multi-turn + tool-calling rollout driver (sv2).

Goal: make the "dataset -> RLHF dataset -> send to rollouts" step explicit, but
for multi-turn/tool-calling via the AgentLoop stack (ToolAgentLoop).

This is a modular harness that supports:
  1) Eval-only mode (train=False): Run rollouts, compute rewards, dump results
  2) Training mode (train=True): Run training loop with periodic eval (PLACEHOLDER)

Modules:
  - sv2/data.py: Data loading utilities
  - sv2/reward.py: GSM8K reward function and reward manager
  - sv2/eval.py: Evaluation (rollouts + reward computation)
  - sv2/train.py: Training loop (placeholder)

Run (eval mode - default):
  python -m sv2.main_ppo_multiturn_toolcall \\
    --config-path sv2/config --config-name sv2_multiturn \\
    data.train_files=$DATA_DIR/train.parquet data.val_files=$DATA_DIR/test.parquet \\
    sv2.dump_jsonl=/tmp/sv2_eval.jsonl

Run (training mode - placeholder):
  python -m sv2.main_ppo_multiturn_toolcall \\
    --config-path sv2/config --config-name sv2_multiturn \\
    train=true \\
    data.train_files=$DATA_DIR/train.parquet data.val_files=$DATA_DIR/test.parquet
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from verl.experimental.agent_loop import AgentLoopManager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

from .data import build_tokenizer_processor, create_dataloader, create_dataset, select_data_paths
from .eval import run_eval
from .train import run_training_loop, TrainConfig


@dataclass(frozen=True)
class Sv2Config:
    """sv2-specific configuration."""

    split: str = "val"  # "train" or "val" for eval-only mode
    batch_size: int = 64
    max_batches: int = -1  # -1 means no limit
    max_samples: int = -1  # -1 means no limit
    dump_jsonl: str | None = None
    interaction_name: str | None = None


def _get_sv2_cfg(config: DictConfig) -> Sv2Config:
    """Extract sv2 config from full config."""
    sv2_cfg = config.get("sv2", {}) or {}
    return Sv2Config(
        split=str(sv2_cfg.get("split", "val")),
        batch_size=int(sv2_cfg.get("batch_size", 64)),
        max_batches=int(sv2_cfg.get("max_batches", -1)),
        max_samples=int(sv2_cfg.get("max_samples", -1)),
        dump_jsonl=sv2_cfg.get("dump_jsonl", None),
        interaction_name=sv2_cfg.get("interaction_name", None),
    )


def _get_train_cfg(config: DictConfig) -> TrainConfig:
    """Extract training config from full config."""
    # Check top-level train flag
    train = config.get("train", False)

    # Get training-specific settings
    train_cfg = config.get("training", {}) or {}

    return TrainConfig(
        train=train,
        total_steps=int(train_cfg.get("total_steps", 100)),
        eval_every_n_steps=int(train_cfg.get("eval_every_n_steps", 10)),
        save_every_n_steps=int(train_cfg.get("save_every_n_steps", 50)),
        ppo_epochs=int(train_cfg.get("ppo_epochs", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-6)),
        clip_ratio=float(train_cfg.get("clip_ratio", 0.2)),
    )


def _init_ray_if_needed(config: DictConfig) -> None:
    """Initialize Ray if not already initialized."""
    if ray.is_initialized():
        return

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"[sv2] ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


def run(config: DictConfig) -> dict[str, Any]:
    """
    Main entry point for sv2.

    Args:
        config: Hydra config

    Returns:
        Results dict with eval/training metrics
    """
    sv2_cfg = _get_sv2_cfg(config)
    train_cfg = _get_train_cfg(config)

    print(f"[sv2] host={socket.gethostname()} pid={os.getpid()}")
    print(f"[sv2] sv2 config: {sv2_cfg}")
    print(f"[sv2] train config: train={train_cfg.train}")

    # Validate config
    if not config.data.get("return_raw_chat", False):
        raise ValueError(
            "This driver requires `data.return_raw_chat=true` so the batch contains `raw_prompt` for AgentLoop."
        )

    if config.actor_rollout_ref.rollout.mode != "async":
        print(
            f"[sv2] Warning: actor_rollout_ref.rollout.mode={config.actor_rollout_ref.rollout.mode!r}; "
            "multi-turn/tool calling typically runs via the async AgentLoop stack."
        )

    # Log mode
    if train_cfg.train:
        print("[sv2] Mode: TRAINING (with periodic eval)")
        print(f"[sv2]   total_steps={train_cfg.total_steps}")
        print(f"[sv2]   eval_every_n_steps={train_cfg.eval_every_n_steps}")
    else:
        print("[sv2] Mode: EVAL ONLY (no weight updates)")

    # Log interaction mode
    multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
    interaction_config_path = getattr(multi_turn_cfg, "interaction_config_path", None)
    if interaction_config_path:
        print(f"[sv2] Interaction mode enabled: config={interaction_config_path}")
        if sv2_cfg.interaction_name:
            print(f"[sv2] Using interaction: {sv2_cfg.interaction_name!r}")

    # Initialize Ray
    _init_ray_if_needed(config)

    # Build tokenizer/processor
    tokenizer, processor = build_tokenizer_processor(config)

    # Create datasets
    if train_cfg.train:
        # Training mode: need both train and val datasets
        train_data_paths = select_data_paths(config, "train")
        val_data_paths = select_data_paths(config, "val")

        train_dataset = create_dataset(
            data_paths=train_data_paths,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            is_train=True,
            max_samples=sv2_cfg.max_samples,
        )
        val_dataset = create_dataset(
            data_paths=val_data_paths,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            is_train=False,
            max_samples=sv2_cfg.max_samples,
        )

        train_dataloader = create_dataloader(
            dataset=train_dataset,
            batch_size=sv2_cfg.batch_size,
            num_workers=int(config.data.get("dataloader_num_workers", 0)),
            shuffle=True,
        )
        val_dataloader = create_dataloader(
            dataset=val_dataset,
            batch_size=sv2_cfg.batch_size,
            num_workers=int(config.data.get("dataloader_num_workers", 0)),
            shuffle=False,
        )
    else:
        # Eval-only mode: just need the specified split
        data_paths = select_data_paths(config, sv2_cfg.split)
        dataset = create_dataset(
            data_paths=data_paths,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            is_train=(sv2_cfg.split == "train"),
            max_samples=sv2_cfg.max_samples,
        )
        val_dataloader = create_dataloader(
            dataset=dataset,
            batch_size=sv2_cfg.batch_size,
            num_workers=int(config.data.get("dataloader_num_workers", 0)),
            shuffle=False,
        )
        train_dataloader = None  # Not needed for eval-only

    # Create AgentLoopManager
    agent_loop_manager = AgentLoopManager(config=config, worker_group=None, rm_resource_pool=None)

    # Run training or eval
    if train_cfg.train:
        # Training mode
        results = run_training_loop(
            agent_loop_manager=agent_loop_manager,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            config=config,
            train_config=train_cfg,
            interaction_name=sv2_cfg.interaction_name,
            dump_dir=sv2_cfg.dump_jsonl.rsplit("/", 1)[0] if sv2_cfg.dump_jsonl else None,
        )
    else:
        # Eval-only mode
        eval_result = run_eval(
            agent_loop_manager=agent_loop_manager,
            dataloader=val_dataloader,
            tokenizer=tokenizer,
            config=config,
            max_batches=sv2_cfg.max_batches,
            num_examine=2,
            interaction_name=sv2_cfg.interaction_name,
            dump_jsonl=sv2_cfg.dump_jsonl,
        )
        results = {
            "mode": "eval_only",
            "num_samples": eval_result.num_samples,
            "mean_reward": eval_result.mean_reward,
            "accuracy": eval_result.accuracy,
            "reward_by_source": eval_result.reward_by_source,
        }

    print(f"[sv2] Results: {results}")
    return results


def _hydra_entrypoint() -> None:
    """Hydra entry point."""
    try:
        import hydra
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency `hydra-core` (import name: `hydra`). "
            "Install it (e.g. `pip install hydra-core`) or run via the normal verl launcher environment."
        ) from e

    @hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def _main(config: DictConfig) -> None:
        run(config)

    _main()


if __name__ == "__main__":
    _hydra_entrypoint()
