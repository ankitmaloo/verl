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
import torch

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

from verl.utils import torch_functional as verl_F
from verl.utils.model import create_huggingface_actor, create_huggingface_critic
from torch.optim import AdamW
from verl.trainer.ppo import core_algos


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


# Global cache for training state
_TRAIN_STATE = {}

def _initialize_train_state(config, tokenizer):
    """Initialize Actor, Critic, Ref, and Optimizers."""
    if _TRAIN_STATE:
        return _TRAIN_STATE

    print("[sv2/train] Initializing Training Components...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Actor Model
    print(f"[sv2/train] Loading Actor: {config.actor_rollout_ref.model.path}")
    actor_module = create_huggingface_actor(
        model_name=config.actor_rollout_ref.model.path,
        override_config_kwargs=config.actor_rollout_ref.model.get("override_config", {}),
        automodel_kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16} 
    )
    actor_module.to(device)
    actor_module.train()
    
    # 2. Reference Model
    print(f"[sv2/train] Loading Ref Policy...")
    ref_module = create_huggingface_actor(
        model_name=config.actor_rollout_ref.model.path,
        override_config_kwargs=config.actor_rollout_ref.model.get("override_config", {}),
        automodel_kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    )
    ref_module.to(device)
    ref_module.eval()
    
    # 3. Critic Model
    # Assumes critic config exists, else fallback
    if hasattr(config, 'critic') and config.critic.model.path:
        critic_path = config.critic.model.path
        override_c = config.critic.model.get("override_config", {})
    else:
        critic_path = config.actor_rollout_ref.model.path # Fallback
        override_c = {}
        
    print(f"[sv2/train] Loading Critic: {critic_path}")
    critic_module = create_huggingface_critic(
        model_name=critic_path,
        override_config_kwargs=override_c,
        automodel_kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    )
    critic_module.to(device)
    critic_module.train()
    
    # 4. Optimizers
    actor_lr = config.actor_rollout_ref.actor.optim.lr
    critic_lr = config.critic.optim.lr if hasattr(config, 'critic') else 1e-5
    
    actor_optim = AdamW(actor_module.parameters(), lr=actor_lr)
    critic_optim = AdamW(critic_module.parameters(), lr=critic_lr)
    
    # Update global state
    _TRAIN_STATE.update({
        "actor": actor_module,
        "ref": ref_module,
        "critic": critic_module,
        "actor_optim": actor_optim,
        "critic_optim": critic_optim,
        "device": device
    })
    return _TRAIN_STATE


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
    Run the training loop with periodic evaluation and Real PPO Updates.
    """
    if not train_config.train:
        print("[sv2/train] train=False, skipping training loop")
        return {"mode": "eval_only"}

    # Training loop
    print(f"[sv2/train] Starting training loop: {train_config.total_steps} steps")
    
    # Initialize Models & Optims
    train_state = _initialize_train_state(config, tokenizer)

    reward_manager = Sv2RewardManager(tokenizer=tokenizer, num_examine=1)
    # Using 'get' with default 1 just in case
    num_rollout_workers = 1
    if hasattr(config.actor_rollout_ref.rollout, "agent"):
         num_rollout_workers = int(config.actor_rollout_ref.rollout.agent.get("num_workers", 1))
         
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
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, num_rollout_workers)

        # Generate rollouts
        # Note: This uses the REMOTE/vLLM agent loop for rollout.
        # Ideally we sync weights from 'actor' to 'vLLM' here, but skipping for this standalone demo.
        print(f"[sv2/train] Step {step}/{train_config.total_steps}: generating rollouts...")
        out_padded = agent_loop_manager.generate_sequences(gen_batch_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        # Merge and compute rewards
        batch_with_output = batch.union(out)
        result = reward_manager(batch_with_output, return_dict=True)
        reward_tensor = result["reward_tensor"]
        mean_reward = reward_tensor.sum(-1).mean().item()

        # PPO update (Real)
        step_result = _ppo_update(
            step=step,
            batch=batch_with_output,
            reward_tensor=reward_tensor,
            config=train_config,
            train_state=train_state,
        )

        history["train_steps"].append({
            "step": step,
            "mean_reward": mean_reward,
            "batch_size": len(batch),
            "loss": step_result.loss,
            "policy_loss": step_result.policy_loss,
            "value_loss": step_result.value_loss,
            "kl_divergence": step_result.kl_divergence,
        })

        print(f"[sv2/train] Step {step}: mean_reward={mean_reward:.4f}, loss={step_result.loss:.4f}")

        # Periodic evaluation
        if train_config.eval_every_n_steps > 0 and step % train_config.eval_every_n_steps == 0:
            print(f"[sv2/train] Running evaluation at step {step}...")
            # Note: eval uses inference weights (vLLM). If we updated local weights, they haven't synced.
            # Real impl needs weight sync.
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
        if train_config.save_every_n_steps > 0 and step % train_config.save_every_n_steps == 0:
            if dump_dir:
                 save_path = f"{dump_dir}/ckpt_{step}"
                 print(f"[sv2/train] Saving checkpoint to {save_path}")
                 train_state["actor"].save_pretrained(save_path)
                 tokenizer.save_pretrained(save_path)

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


def _ppo_update(
    step: int,
    batch: DataProto,
    reward_tensor: torch.Tensor,
    config: TrainConfig,
    train_state: dict
) -> TrainStepResult:
    """
    Perform PPO update step using local Actor/Critic/Ref models.
    """
    actor = train_state["actor"]
    critic = train_state["critic"]
    ref_policy = train_state["ref"]
    actor_optim = train_state["actor_optim"]
    critic_optim = train_state["critic_optim"]
    device = train_state["device"]

    # Move batch to device keys needed
    # Verl DataProto is cleaner, but here we assume standard HF keys in batch
    
    # We need to construct tensors from DataProto
    # DataProto batch keys: input_ids (list->tensor), responses (list->tensor)
    # They are lists of ints usually in batch, need stacking
    
    input_ids_list = batch.batch["input_ids"]
    responses_list = batch.batch["responses"]
    
    # Simple stacking assuming padding was handled or uniform
    # In verl, they are usually tensors already if we used default collate
    input_ids = batch.batch["input_ids"].to(device)
    response_ids = batch.batch["responses"].to(device) 
    attention_mask = batch.batch["attention_mask"].to(device)

    # For PPO, we usually concatenate Prompt + Response for full forward pass
    # But batch keys might be separated.
    # Verl's `generate_sequences` returns full sequences usually? 
    # Actually `agent_loop_manager` returns responses. input_ids is prompt.
    # We need to check if we need to concat.
    # Usually: Full Seq = [Prompt, Response]
    
    # Let's assume input_ids is the Prompt. response_ids is the Completion.
    # We construct full_ids for forward passing the causal LM.
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    full_attention_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1) # naive mask
    
    response_length = response_ids.shape[-1]
    response_mask = (response_ids != 0).float() # Assuming 0 is pad check
    
    # 1. Compute Values & Ref LogProbs (No Grad)
    with torch.no_grad():
        # Values (Critic)
        # Critic typically expects full sequence but we care about values for Response tokens
        critic_out = critic(input_ids=full_ids, attention_mask=full_attention_mask)
        # logits shape (B, Seq, 1). We squeeze.
        values = critic_out.logits.squeeze(-1)
        # Slice to get value estimates for the response part
        # Values[t] is usually V(s_t).
        # We need values matching tokens. 
        values = values[:, -response_length:]
        
        # Ref LogProbs
        ref_out = ref_policy(input_ids=full_ids, attention_mask=full_attention_mask)
        ref_logits = ref_out.logits[:, :-1] # remove last prediction
        # We align with full_ids[:, 1:]
        # We only care about response part.
        # Response starts at index: input_ids.shape[1] - 1 (in logits)?
        prompt_len = input_ids.shape[1]
        
        # Slice logits corresponding to response tokens
        # Target tokens: response_ids
        # Logits needed: indices [prompt_len-1 : -1] ??
        # Let's say seq is [p1, p2, r1, r2].
        # Logits: [L(p1), L(p2), L(r1), L(r2)]
        # We need P(r1|p1,p2) -> L(p2). P(r2|...) -> L(r1).
        # So we take logits from index (prompt_len - 1) up to (total - 1).
        
        ref_resp_logits = ref_logits[:, prompt_len-1 : ]
        ref_log_probs = verl_F.log_probs_from_logits(ref_resp_logits, response_ids)

        # Old LogProbs
        # Ideally passed in `gen_batch["response_logprobs"]`
        if "response_logprobs" in batch.batch:
             old_log_probs = batch.batch["response_logprobs"].to(device)
        else:
             # Recompute if missing using Actor (approximate if actor changed, but here we are step 0)
             actor_out = actor(input_ids=full_ids, attention_mask=full_attention_mask)
             actor_resp_logits = actor_out.logits[:, prompt_len-1 : -1]
             old_log_probs = verl_F.log_probs_from_logits(actor_resp_logits, response_ids)
             
    # 2. Rewards & Advantages
    # Add KL penalty to rewards
    # Reward tensor is usually final outcome score (scalar per sequence) or sparse.
    # We will assume reward_tensor is (B,) outcome score.
    # We construct token_level_rewards: 0 everywhere, reward at last token.
    token_level_rewards = torch.zeros_like(old_log_probs)
    
    # Scatter reward to last valid token
    # Simple approx: last token. logic needs mask.
    # For now, put at last index.
    for i in range(len(token_level_rewards)):
        # find length
        l = int(response_mask[i].sum().item())
        if l > 0:
            token_level_rewards[i, l-1] += reward_tensor[i]
            
    # KL Penalty
    kl = old_log_probs - ref_log_probs
    # token_level_rewards -= 0.01 * kl # beta hardcoded for now
    
    # GAE
    with torch.no_grad():
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma, # using user config
            lam=config.gae_lambda
        )

    # 3. PPO Epochs
    total_p_loss = 0.0
    total_v_loss = 0.0
    
    for _ in range(config.ppo_epochs):
        # Forward Actor
        actor_out = actor(input_ids=full_ids, attention_mask=full_attention_mask)
        curr_logits = actor_out.logits[:, prompt_len-1 : -1]
        curr_log_probs = verl_F.log_probs_from_logits(curr_logits, response_ids)
        
        # Forward Critic
        critic_out = critic(input_ids=full_ids, attention_mask=full_attention_mask)
        curr_values = critic_out.logits.squeeze(-1)[:, -response_length:]
        
        # Losses
        pg_loss, pg_metrics, _, _ = core_algos.compute_policy_loss(
            old_log_prob=old_log_probs,
            log_prob=curr_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=config.clip_ratio,
            loss_agg_mode="token-mean"
        )
        
        vf_loss, _ = core_algos.compute_value_loss(
            vpreds=curr_values,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="token-mean"
        )
        
        loss = pg_loss + 0.1 * vf_loss
        
        actor_optim.zero_grad()
        critic_optim.zero_grad()
        loss.backward()
        actor_optim.step()
        critic_optim.step()
        
        total_p_loss += pg_loss.item()
        total_v_loss += vf_loss.item()
        
    return TrainStepResult(
        step=step,
        batch_size=len(batch),
        mean_reward=reward_tensor.sum(-1).mean().item(),
        loss=(total_p_loss + total_v_loss)/config.ppo_epochs,
        policy_loss=total_p_loss/config.ppo_epochs,
        value_loss=total_v_loss/config.ppo_epochs,
        kl_divergence=pg_metrics.get("actor/ppo_kl", 0.0)
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
