"""
sv2/train.py - Training loop placeholder for multi-turn RL.

This is a PLACEHOLDER for the training loop. When `train=True` in config,
this module would handle:
1. PPO/GRPO training steps
2. Weight updates
3. Periodic evaluation

Currently implements:
- Training loop structure with eval_every_n_steps
- Placeholder for PPO update (prints warning, no actual update)
- Calls eval.run_eval for periodic evaluation

To enable actual training, implement the `_ppo_update` function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Support both running as package and standalone
try:
    from .eval import run_eval, EvalResult
    from .reward import Sv2RewardManager
except ImportError:
    from eval import run_eval, EvalResult
    from reward import Sv2RewardManager

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizer

    from verl.experimental.agent_loop import AgentLoopManager


@dataclass
class TrainStepResult:
    """Result from a single training step."""

    step: int
    batch_size: int
    mean_reward: float
    # Placeholder for training metrics
    loss: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    kl_divergence: float | None = None


@dataclass
class TrainConfig:
    """Training configuration."""

    train: bool = False  # Master flag - if False, skip all training
    total_steps: int = 100
    eval_every_n_steps: int = 10
    save_every_n_steps: int = 50
    # PPO hyperparams (placeholders)
    ppo_epochs: int = 1
    learning_rate: float = 1e-6
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95


def run_training_loop(
    agent_loop_manager: AgentLoopManager,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    config,
    train_config: TrainConfig,
    interaction_name: str | None = None,
    dump_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run the training loop with periodic evaluation.

    This is a PLACEHOLDER implementation. The actual PPO update is not implemented.

    Args:
        agent_loop_manager: AgentLoopManager for generating sequences
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        tokenizer: Tokenizer for decoding
        config: Full config
        train_config: Training-specific config
        interaction_name: Override interaction name
        dump_dir: Directory to dump results

    Returns:
        Dict with training history
    """
    if not train_config.train:
        print("[sv2/train] train=False, skipping training loop")
        print("[sv2/train] Running eval-only mode...")
        eval_result = run_eval(
            agent_loop_manager=agent_loop_manager,
            dataloader=val_dataloader,
            tokenizer=tokenizer,
            config=config,
            max_batches=-1,
            num_examine=2,
            interaction_name=interaction_name,
            dump_jsonl=f"{dump_dir}/eval_only.jsonl" if dump_dir else None,
        )
        return {
            "mode": "eval_only",
            "eval_result": {
                "num_samples": eval_result.num_samples,
                "mean_reward": eval_result.mean_reward,
                "accuracy": eval_result.accuracy,
            },
        }

    # Training loop
    print(f"[sv2/train] Starting training loop: {train_config.total_steps} steps")
    print(f"[sv2/train] WARNING: PPO update is a PLACEHOLDER - no actual weight updates!")

    reward_manager = Sv2RewardManager(tokenizer=tokenizer, num_examine=1)
    num_workers = int(config.actor_rollout_ref.rollout.agent.num_workers)
    interaction_config_path = getattr(
        config.actor_rollout_ref.rollout.multi_turn, "interaction_config_path", None
    )

    history = {
        "train_steps": [],
        "eval_results": [],
    }

    train_iter = iter(train_dataloader)
    step = 0

    while step < train_config.total_steps:
        # Get next batch (cycle through dataloader)
        try:
            batch_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch_dict = next(train_iter)

        step += 1

        # Convert to DataProto
        batch = DataProto.from_single_dict(
            batch_dict,
            meta_info={"global_steps": step, "validate": False},
        )

        # Add UIDs
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [f"train_{step}_{i}" for i in range(len(batch))],
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
        gen_batch = _get_gen_batch(batch)

        # Pad for divisibility
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, num_workers)

        # Generate rollouts
        print(f"[sv2/train] Step {step}/{train_config.total_steps}: generating rollouts...")
        out_padded = agent_loop_manager.generate_sequences(gen_batch_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        # Merge and compute rewards
        batch_with_output = batch.union(out)
        result = reward_manager(batch_with_output, return_dict=True)
        reward_tensor = result["reward_tensor"]
        mean_reward = reward_tensor.sum(-1).mean().item()

        # PLACEHOLDER: PPO update would happen here
        step_result = _ppo_update_placeholder(
            step=step,
            batch=batch_with_output,
            reward_tensor=reward_tensor,
            config=train_config,
        )

        history["train_steps"].append({
            "step": step,
            "mean_reward": mean_reward,
            "batch_size": len(batch),
        })

        print(f"[sv2/train] Step {step}: mean_reward={mean_reward:.4f}")

        # Periodic evaluation
        if step % train_config.eval_every_n_steps == 0:
            print(f"[sv2/train] Running evaluation at step {step}...")
            eval_result = run_eval(
                agent_loop_manager=agent_loop_manager,
                dataloader=val_dataloader,
                tokenizer=tokenizer,
                config=config,
                max_batches=5,  # Limit eval batches during training
                num_examine=1,
                interaction_name=interaction_name,
                dump_jsonl=f"{dump_dir}/eval_step_{step}.jsonl" if dump_dir else None,
            )
            history["eval_results"].append({
                "step": step,
                "num_samples": eval_result.num_samples,
                "mean_reward": eval_result.mean_reward,
                "accuracy": eval_result.accuracy,
            })
            print(f"[sv2/train] Eval at step {step}: "
                  f"mean_reward={eval_result.mean_reward:.4f}, "
                  f"accuracy={eval_result.accuracy:.4f}")

        # Periodic save (placeholder)
        if step % train_config.save_every_n_steps == 0:
            print(f"[sv2/train] PLACEHOLDER: Would save checkpoint at step {step}")

    # Final evaluation
    print("[sv2/train] Running final evaluation...")
    final_eval = run_eval(
        agent_loop_manager=agent_loop_manager,
        dataloader=val_dataloader,
        tokenizer=tokenizer,
        config=config,
        max_batches=-1,
        num_examine=2,
        interaction_name=interaction_name,
        dump_jsonl=f"{dump_dir}/eval_final.jsonl" if dump_dir else None,
    )
    history["final_eval"] = {
        "num_samples": final_eval.num_samples,
        "mean_reward": final_eval.mean_reward,
        "accuracy": final_eval.accuracy,
    }

    return history


def _ppo_update_placeholder(
    step: int,
    batch: DataProto,
    reward_tensor,
    config: TrainConfig,
) -> TrainStepResult:
    """
    PLACEHOLDER for PPO update.

    When implementing actual training, this function should:
    1. Compute advantages (GAE)
    2. Compute old log probs
    3. Run PPO epochs:
       - Compute new log probs
       - Compute policy loss with clipping
       - Compute value loss
       - Update actor and critic

    For now, just returns dummy metrics.
    """
    # This is where the actual PPO update would happen
    # For now, we just log a warning and return dummy values
    if step == 1:
        print("[sv2/train] WARNING: _ppo_update_placeholder called - no actual weight update!")
        print("[sv2/train] To enable training, implement _ppo_update in sv2/train.py")

    return TrainStepResult(
        step=step,
        batch_size=len(batch),
        mean_reward=reward_tensor.sum(-1).mean().item(),
        loss=None,  # Placeholder
        policy_loss=None,
        value_loss=None,
        kl_divergence=None,
    )


def _get_gen_batch(batch: DataProto) -> DataProto:
    """Prepare a generation batch from the full batch."""
    reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"} & batch.non_tensor_batch.keys()

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys

    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=list(non_tensor_keys_to_pop),
    )

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
