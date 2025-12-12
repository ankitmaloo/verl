"""
sv2/eval.py - Evaluation for multi-turn rollouts.

Runs rollouts on validation data and computes rewards.
This is the "eval" part of the training loop - no weight updates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from .reward import Sv2RewardManager

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizer

    from verl.experimental.agent_loop import AgentLoopManager


@dataclass
class EvalResult:
    """Results from an evaluation run."""

    num_samples: int
    mean_reward: float
    reward_by_source: dict[str, float]
    accuracy: float  # Fraction of samples with reward > 0
    samples: list[dict[str, Any]]  # Individual sample results
    extra_info: dict[str, Any]


def run_eval(
    agent_loop_manager: AgentLoopManager,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    config,
    max_batches: int = -1,
    num_examine: int = 1,
    interaction_name: str | None = None,
    dump_jsonl: str | None = None,
) -> EvalResult:
    """
    Run evaluation: generate rollouts and compute rewards.

    Args:
        agent_loop_manager: AgentLoopManager for generating sequences
        dataloader: DataLoader yielding batches
        tokenizer: Tokenizer for decoding
        config: Full config (for accessing rollout settings)
        max_batches: Max batches to evaluate (-1 for all)
        num_examine: Number of samples to print per data source
        interaction_name: Override interaction name for all samples
        dump_jsonl: Path to dump results as JSONL

    Returns:
        EvalResult with metrics and samples
    """
    reward_manager = Sv2RewardManager(tokenizer=tokenizer, num_examine=num_examine)

    all_samples = []
    all_rewards = []
    reward_by_source = defaultdict(list)

    num_workers = int(config.actor_rollout_ref.rollout.agent.num_workers)
    interaction_config_path = getattr(
        config.actor_rollout_ref.rollout.multi_turn, "interaction_config_path", None
    )

    for batch_idx, batch_dict in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        # Convert to DataProto
        batch = DataProto.from_single_dict(
            batch_dict,
            meta_info={"global_steps": 0, "validate": True},
        )

        # Add UIDs if missing
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [f"eval_{batch_idx}_{i}" for i in range(len(batch))],
                dtype=object,
            )

        # Set agent_name if missing
        if "agent_name" not in batch.non_tensor_batch:
            multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
            has_tools = getattr(multi_turn_cfg, "tool_config_path", None)
            has_interaction = getattr(multi_turn_cfg, "interaction_config_path", None)
            agent_name = "tool_agent" if (has_tools or has_interaction) else "single_turn_agent"
            batch.non_tensor_batch["agent_name"] = np.array([agent_name] * len(batch), dtype=object)

        # Inject interaction_kwargs if needed
        if interaction_config_path and interaction_name:
            _ensure_interaction_kwargs(batch, interaction_name)

        # Prepare generation batch
        gen_batch = _get_gen_batch_for_eval(batch)

        # Pad for divisibility
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, num_workers)

        # Generate sequences
        print(f"[sv2/eval] Generating batch {batch_idx}, size={len(gen_batch)}")
        out_padded = agent_loop_manager.generate_sequences(gen_batch_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        # Merge output with original batch for reward computation
        batch_with_output = batch.union(out)
        batch_with_output.meta_info["validate"] = True

        # Compute rewards
        result = reward_manager(batch_with_output, return_dict=True)
        reward_tensor = result["reward_tensor"]
        scores = reward_tensor.sum(-1).cpu().tolist()
        all_rewards.extend(scores)

        # Collect per-source rewards
        data_sources = batch_with_output.non_tensor_batch.get("data_source", ["unknown"] * len(batch))
        for i, (score, source) in enumerate(zip(scores, data_sources)):
            reward_by_source[source].append(score)

        # Decode responses for logging
        responses = _decode_responses(out, tokenizer)
        prompts = _decode_prompts(batch, tokenizer)
        num_turns = out.non_tensor_batch.get("__num_turns__")
        uids = out.non_tensor_batch.get("uid")
        ground_truths = [
            item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            for item in batch_with_output
        ]

        # Collect samples
        for i in range(len(out)):
            sample = {
                "uid": uids[i] if uids is not None else None,
                "prompt": prompts[i],
                "response": responses[i],
                "ground_truth": ground_truths[i],
                "score": scores[i],
                "num_turns": int(num_turns[i]) if num_turns is not None else None,
                "data_source": data_sources[i] if data_sources is not None else "unknown",
            }
            all_samples.append(sample)

        print(f"[sv2/eval] Batch {batch_idx}: mean_reward={np.mean(scores):.3f}")

    # Compute aggregate metrics
    mean_reward = np.mean(all_rewards) if all_rewards else 0.0
    accuracy = np.mean([1.0 if r > 0 else 0.0 for r in all_rewards]) if all_rewards else 0.0

    source_means = {
        source: np.mean(rewards) for source, rewards in reward_by_source.items()
    }

    # Dump to JSONL if requested
    if dump_jsonl:
        _dump_jsonl(dump_jsonl, all_samples)
        print(f"[sv2/eval] Dumped {len(all_samples)} samples to {dump_jsonl}")

    return EvalResult(
        num_samples=len(all_samples),
        mean_reward=mean_reward,
        reward_by_source=source_means,
        accuracy=accuracy,
        samples=all_samples,
        extra_info={},
    )


def _get_gen_batch_for_eval(batch: DataProto) -> DataProto:
    """Prepare a generation batch from the full batch."""
    reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"} & batch.non_tensor_batch.keys()

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys

    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=list(non_tensor_keys_to_pop),
    )

    # AgentLoop needs all non-tensor fields
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    return gen_batch


def _ensure_interaction_kwargs(batch: DataProto, interaction_name: str) -> None:
    """Ensure each sample has interaction_kwargs in extra_info."""
    batch_size = len(batch)
    extra_info = batch.non_tensor_batch.get("extra_info")

    if extra_info is None:
        extra_info = np.array(
            [{"interaction_kwargs": {"name": interaction_name}} for _ in range(batch_size)],
            dtype=object,
        )
        batch.non_tensor_batch["extra_info"] = extra_info
        return

    for i in range(batch_size):
        if extra_info[i] is None:
            extra_info[i] = {}
        if "interaction_kwargs" not in extra_info[i]:
            extra_info[i]["interaction_kwargs"] = {"name": interaction_name}
        elif "name" not in extra_info[i]["interaction_kwargs"]:
            extra_info[i]["interaction_kwargs"]["name"] = interaction_name


def _decode_responses(output: DataProto, tokenizer) -> list[str]:
    """Decode response tokens to strings."""
    prompt_len = output.batch["prompts"].shape[1]
    decoded = []
    for i in range(len(output)):
        resp_len = int(output.batch["attention_mask"][i, prompt_len:].sum().item())
        resp_ids = output.batch["responses"][i, :resp_len].tolist()
        decoded.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
    return decoded


def _decode_prompts(batch: DataProto, tokenizer) -> list[str]:
    """Decode prompt tokens to strings."""
    decoded = []
    for i in range(len(batch)):
        input_ids = batch.batch["input_ids"][i]
        attn_mask = batch.batch["attention_mask"][i]
        valid_len = attn_mask.sum().item()
        valid_ids = input_ids[-int(valid_len):].tolist()
        decoded.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
    return decoded


def _dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Dump rows to a JSONL file."""
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
