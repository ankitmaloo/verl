"""
Multi-turn + tool-calling rollout driver (sv2).

Goal: make the "dataset -> RLHF dataset -> send to rollouts" step explicit, but
for multi-turn/tool-calling via the AgentLoop stack (ToolAgentLoop).

This is intentionally *not* a full PPO trainer. It is a small harness to:
  1) Load a verl-format RLHF parquet dataset (or custom dataset class via config)
  2) Build RLHFDataset items (tokenized prompt + raw chat)
  3) Convert a batch to DataProto and strip it to a generation batch
  4) Send the batch to AgentLoopManager.generate_sequences (multi-turn/tool calling)
  5) Decode and optionally dump results

Supports two modes:
  A) Tool calling: Model calls tools (calculator, search, etc.) which get executed
  B) Interaction: After model response, inject user feedback and continue generation

Run (example with interaction - user asks for code verification):
  python -m sv2.main_ppo_multiturn_toolcall \\
    --config-path sv2/config --config-name sv2_multiturn \\
    data.train_files=$DATA_DIR/train.parquet data.val_files=$DATA_DIR/test.parquet \\
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=sv2/config/interaction_config.yaml \\
    sv2.interaction_name=code_verify sv2.batch_size=4 sv2.max_batches=1

Notes:
  - Tool calling is chosen via per-sample `agent_name` (e.g. `tool_agent`), which
    should be present in the parquet (see `examples/data_preprocess/gsm8k_tool_agent_loop.py`).
  - For "user adds a string and it keeps going" (Interaction mode), set:
    - `actor_rollout_ref.rollout.multi_turn.interaction_config_path=...`
    - `sv2.interaction_name=<name>` (or have `extra_info.interaction_kwargs.name` in dataset)
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local


@dataclass(frozen=True)
class Sv2RolloutConfig:
    split: str = "val"  # "train" or "val"
    batch_size: int = 4
    max_batches: int = 1
    max_samples: int = -1  # -1 means no limit; applied at dataset construction
    dump_jsonl: str | None = None
    # Interaction mode: name of the interaction to use (from interaction_config.yaml)
    # If set, injects interaction_kwargs into samples that don't have it
    interaction_name: str | None = None


def _get_sv2_cfg(config: DictConfig) -> Sv2RolloutConfig:
    sv2_cfg = config.get("sv2", {}) or {}
    return Sv2RolloutConfig(
        split=str(sv2_cfg.get("split", "val")),
        batch_size=int(sv2_cfg.get("batch_size", 4)),
        max_batches=int(sv2_cfg.get("max_batches", 1)),
        max_samples=int(sv2_cfg.get("max_samples", -1)),
        dump_jsonl=sv2_cfg.get("dump_jsonl", None),
        interaction_name=sv2_cfg.get("interaction_name", None),
    )


def _init_ray_if_needed(config: DictConfig) -> None:
    if ray.is_initialized():
        return

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"[sv2] ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


def _build_tokenizer_processor(config: DictConfig):
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    return tokenizer, processor


def _select_data_paths(config: DictConfig, split: str) -> Any:
    if split == "train":
        return config.data.train_files
    if split == "val":
        return config.data.val_files
    raise ValueError(f"sv2.split must be 'train' or 'val', got {split!r}")


def _get_gen_batch_for_agent_loop(batch: DataProto) -> DataProto:
    reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys

    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
    )

    # AgentLoop needs all non-tensor fields (raw_prompt, tools_kwargs, agent_name, ...)
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    return gen_batch


def _decode_rollout_responses(output: DataProto, tokenizer) -> list[str]:
    prompt_len = output.batch["prompts"].shape[1]
    decoded: list[str] = []
    for i in range(len(output)):
        resp_len = int(output.batch["attention_mask"][i, prompt_len:].sum().item())
        resp_ids = output.batch["responses"][i, :resp_len].tolist()
        decoded.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
    return decoded


def _dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_interaction_kwargs(
    batch: DataProto,
    interaction_name: str | None,
    interaction_config_path: str | None,
) -> None:
    """
    Ensure each sample has interaction_kwargs in extra_info if interaction is enabled.

    This allows the driver to enable interaction mode even if the dataset
    doesn't have interaction_kwargs pre-baked.
    """
    if not interaction_config_path or not interaction_name:
        return

    batch_size = len(batch)
    extra_info = batch.non_tensor_batch.get("extra_info")

    if extra_info is None:
        # Create extra_info array with interaction_kwargs
        extra_info = np.array(
            [{"interaction_kwargs": {"name": interaction_name}} for _ in range(batch_size)],
            dtype=object,
        )
        batch.non_tensor_batch["extra_info"] = extra_info
        print(f"[sv2] Injected interaction_kwargs.name={interaction_name!r} into {batch_size} samples (new extra_info)")
        return

    # extra_info exists, ensure each has interaction_kwargs
    modified_count = 0
    for i in range(batch_size):
        if extra_info[i] is None:
            extra_info[i] = {}

        if "interaction_kwargs" not in extra_info[i]:
            extra_info[i]["interaction_kwargs"] = {"name": interaction_name}
            modified_count += 1
        elif "name" not in extra_info[i]["interaction_kwargs"]:
            extra_info[i]["interaction_kwargs"]["name"] = interaction_name
            modified_count += 1

    if modified_count > 0:
        print(f"[sv2] Injected interaction_kwargs.name={interaction_name!r} into {modified_count}/{batch_size} samples")

def run(config: DictConfig) -> None:
    sv2_cfg = _get_sv2_cfg(config)
    print(f"[sv2] host={socket.gethostname()} pid={os.getpid()}")
    print(f"[sv2] cfg: {sv2_cfg}")

    if not config.data.get("return_raw_chat", False):
        raise ValueError(
            "This driver requires `data.return_raw_chat=true` so the batch contains `raw_prompt` for AgentLoop."
        )

    if config.actor_rollout_ref.rollout.mode != "async":
        print(
            f"[sv2] Warning: actor_rollout_ref.rollout.mode={config.actor_rollout_ref.rollout.mode!r}; "
            "multi-turn/tool calling typically runs via the async AgentLoop stack."
        )

    # Log interaction mode status
    multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
    interaction_config_path = getattr(multi_turn_cfg, "interaction_config_path", None)
    if interaction_config_path:
        print(f"[sv2] Interaction mode enabled: config={interaction_config_path}")
        if sv2_cfg.interaction_name:
            print(f"[sv2] Using interaction: {sv2_cfg.interaction_name!r}")
        else:
            print("[sv2] Warning: interaction_config_path set but sv2.interaction_name not set. "
                  "Dataset must have extra_info.interaction_kwargs.name for each sample.")

    _init_ray_if_needed(config)
    tokenizer, processor = _build_tokenizer_processor(config)

    data_paths = _select_data_paths(config, sv2_cfg.split)
    dataset = create_rl_dataset(
        data_paths=data_paths,
        data_config=config.data,
        tokenizer=tokenizer,
        processor=processor,
        is_train=(sv2_cfg.split == "train"),
        max_samples=sv2_cfg.max_samples,
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=sv2_cfg.batch_size,
        shuffle=False,
        num_workers=int(config.data.get("dataloader_num_workers", 0)),
        collate_fn=collate_fn,
    )

    agent_loop_manager = AgentLoopManager(config=config, worker_group=None, rm_resource_pool=None)

    dumped_rows: list[dict[str, Any]] = []
    for batch_idx, batch_dict in enumerate(dataloader):
        if batch_idx >= sv2_cfg.max_batches:
            break

        batch = DataProto.from_single_dict(batch_dict, meta_info={"global_steps": 0, "validate": sv2_cfg.split == "val"})

        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array([f"sv2_{batch_idx}_{i}" for i in range(len(batch))], dtype=object)

        if "agent_name" not in batch.non_tensor_batch:
            # For interaction mode, we still need tool_agent (it handles both tools and interactions)
            multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
            has_tools = getattr(multi_turn_cfg, "tool_config_path", None)
            has_interaction = getattr(multi_turn_cfg, "interaction_config_path", None)
            if has_tools or has_interaction:
                inferred = "tool_agent"
            else:
                inferred = "single_turn_agent"
            print(
                f"[sv2] Warning: dataset has no `agent_name`; defaulting to {inferred!r}. "
                "For tool calling or interactions, set parquet column agent_name='tool_agent'."
            )
            batch.non_tensor_batch["agent_name"] = np.array([inferred] * len(batch), dtype=object)

        # Inject interaction_kwargs if interaction mode is enabled
        interaction_config_path = getattr(
            config.actor_rollout_ref.rollout.multi_turn, "interaction_config_path", None
        )
        _ensure_interaction_kwargs(batch, sv2_cfg.interaction_name, interaction_config_path)

        gen_batch = _get_gen_batch_for_agent_loop(batch)
        size_divisor = int(config.actor_rollout_ref.rollout.agent.num_workers)
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        out_padded = agent_loop_manager.generate_sequences(gen_batch_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        decoded = _decode_rollout_responses(out, tokenizer)
        num_turns = out.non_tensor_batch.get("__num_turns__")
        uids = out.non_tensor_batch.get("uid")

        print(f"[sv2] batch {batch_idx}: size={len(out)}")
        for i in range(min(len(out), 2)):
            turns_i = int(num_turns[i]) if num_turns is not None else None
            print(f"[sv2] sample[{i}] uid={uids[i] if uids is not None else None} turns={turns_i}")
            print(decoded[i][:1000])

        if sv2_cfg.dump_jsonl:
            raw_prompts = out.non_tensor_batch.get("raw_prompt")
            for i in range(len(out)):
                dumped_rows.append(
                    {
                        "uid": None if uids is None else uids[i],
                        "num_turns": None if num_turns is None else int(num_turns[i]),
                        "raw_prompt": None if raw_prompts is None else raw_prompts[i],
                        "decoded_response": decoded[i],
                        "tool_extra_fields": out.non_tensor_batch.get("tool_extra_fields", None)[i]
                        if "tool_extra_fields" in out.non_tensor_batch
                        else None,
                    }
                )

    if sv2_cfg.dump_jsonl:
        _dump_jsonl(sv2_cfg.dump_jsonl, dumped_rows)
        print(f"[sv2] wrote {len(dumped_rows)} rows -> {sv2_cfg.dump_jsonl}")


def _hydra_entrypoint() -> None:
    try:
        import hydra  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
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
