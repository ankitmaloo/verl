"""VERL + SGLang async inference harness using SGLangRollout.

Supports nested config (rollout.*) and flat config (backwards compatible).
Uses VERL's SGLangRollout in async mode for optimal performance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_visible_devices_keyword
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

Message = Dict[str, Any]


@dataclass
class GenerationOutput:
    completions: List[str]
    token_ids: List[List[int]]
    metadata: Optional[List[Dict[str, Any]]] = None


def _ensure_dist_initialized():
    import torch.distributed as dist

    if dist.is_initialized():
        return

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    devices_keyword = get_visible_devices_keyword()
    os.environ.setdefault(devices_keyword, "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


class InferenceEngine:
    """Lightweight wrapper that reuses VERL's SGLangRollout end-to-end."""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        _ensure_dist_initialized()

        self.model_config = HFModelConfig(
            path=self.config["model_path"],
            trust_remote_code=self.config.get("trust_remote_code", False),
        )
        self.tokenizer = self.model_config.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.processor = self.model_config.get_processor()

        # Set environment variable required by SGLang for tensor parallelism
        os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "True")

        rollout_conf = self._build_rollout_config()
        self.rollout = SGLangRollout(
            config=rollout_conf,
            model_config=self.model_config,
            device_mesh=None,
        )
        # Smoke test manually if needed: `InferenceEngine(cfg).generate(["what is 2+2?"])`.

    def _build_rollout_config(self) -> RolloutConfig:
        """Build RolloutConfig from nested config structure."""
        # Get rollout config (supports both nested and flat config for backwards compatibility)
        rollout_cfg = self.config.get("rollout", {})
        
        # If rollout key doesn't exist, use flat config (backwards compatible)
        if not rollout_cfg:
            rollout_cfg = self.config
        
        # Store resolved prompt/response lengths for consistent use
        self._prompt_length = rollout_cfg.get("prompt_length", self.config.get("max_prompt_length", 2048))
        self._response_length = rollout_cfg.get("response_length", self.config.get("max_response_length", 512))
        
        rollout_dict = {
            # Core async mode settings (REQUIRED)
            "name": rollout_cfg.get("name", "sglang"),
            "mode": rollout_cfg.get("mode", "async"),  # Default to async!
            "skip_tokenizer_init": rollout_cfg.get("skip_tokenizer_init", True),
            
            # Sequence length settings
            "prompt_length": self._prompt_length,
            "response_length": self._response_length,
            "max_model_len": rollout_cfg.get("max_model_len"),
            "max_num_seqs": rollout_cfg.get("max_num_seqs", 1024),
            "max_num_batched_tokens": rollout_cfg.get("max_num_batched_tokens", 8192),
            
            # Sampling parameters
            "temperature": rollout_cfg.get("temperature", self.config.get("temperature", 1.0)),
            "top_p": rollout_cfg.get("top_p", self.config.get("top_p", 1.0)),
            "top_k": rollout_cfg.get("top_k", self.config.get("top_k", -1)),
            "do_sample": rollout_cfg.get("do_sample", self.config.get("do_sample", True)),
            "n": rollout_cfg.get("n", self.config.get("n", 1)),
            "ignore_eos": rollout_cfg.get("ignore_eos", False),
            
            # Validation sampling
            "val_kwargs": rollout_cfg.get("val_kwargs", {
                "temperature": 0,
                "top_k": -1,
                "top_p": 1.0,
                "do_sample": False,
                "n": 1,
            }),
            
            # GPU & parallelism settings
            "tensor_model_parallel_size": rollout_cfg.get(
                "tensor_model_parallel_size", 
                self.config.get("num_gpus", 1)
            ),
            "data_parallel_size": rollout_cfg.get("data_parallel_size", 1),
            "expert_parallel_size": rollout_cfg.get("expert_parallel_size", 1),
            "pipeline_model_parallel_size": rollout_cfg.get("pipeline_model_parallel_size", 1),
            
            # Memory & performance
            "gpu_memory_utilization": rollout_cfg.get(
                "gpu_memory_utilization",
                self.config.get("gpu_memory_utilization", 0.5)
            ),
            "dtype": rollout_cfg.get("dtype", self.config.get("dtype", "bfloat16")),
            "free_cache_engine": rollout_cfg.get("free_cache_engine", True),
            "enforce_eager": rollout_cfg.get("enforce_eager", False),
            "cudagraph_capture_sizes": rollout_cfg.get("cudagraph_capture_sizes"),
            
            # Performance optimizations
            "enable_chunked_prefill": rollout_cfg.get("enable_chunked_prefill", True),
            "enable_prefix_caching": rollout_cfg.get("enable_prefix_caching", True),
            
            # Model loading - IMPORTANT: default to "auto" for real weights!
            "load_format": rollout_cfg.get("load_format", "auto"),
            
            # SGLang engine kwargs
            "engine_kwargs": rollout_cfg.get("engine_kwargs", {}),
            
            # Logging & debugging
            "calculate_log_probs": rollout_cfg.get(
                "calculate_log_probs",
                self.config.get("return_logprobs", False)
            ),
            "disable_log_stats": rollout_cfg.get("disable_log_stats", True),
            
            # Advanced settings
            "over_sample_rate": rollout_cfg.get("over_sample_rate", 0.0),
            "update_weights_bucket_megabytes": rollout_cfg.get("update_weights_bucket_megabytes", 512),
            "multi_stage_wake_up": rollout_cfg.get("multi_stage_wake_up", False),
            
            # Multi-turn configuration
            "multi_turn": self._build_multi_turn_cfg(),
            
            # Caching & debugging
            "skip_rollout": rollout_cfg.get("skip_rollout", False),
            "skip_dump_dir": rollout_cfg.get("skip_dump_dir", "/tmp/rollout_dump"),
            
            # Profiling
            "profiler": rollout_cfg.get("profiler"),
        }

        omega = OmegaConf.create(rollout_dict)
        return omega_conf_to_dataclass(omega, dataclass_type=RolloutConfig)

    def _build_multi_turn_cfg(self) -> Dict[str, Any]:
        """Build multi-turn config from nested config structure."""
        # Try to get from rollout.multi_turn first, fall back to flat multi_turn
        rollout_cfg = self.config.get("rollout", {})
        if rollout_cfg and "multi_turn" in rollout_cfg:
            mt = rollout_cfg.get("multi_turn", {}) or {}
        else:
            mt = self.config.get("multi_turn", {}) or {}
        
        return {
            "enable": mt.get("enable", False),
            "max_assistant_turns": mt.get("max_assistant_turns", mt.get("max_turns")),
            "max_user_turns": mt.get("max_user_turns", mt.get("max_turns")),
            "max_parallel_calls": mt.get("max_parallel_calls", 1),
            "max_tool_response_length": mt.get(
                "max_tool_response_length", self.config.get("max_response_length", 512)
            ),
            "tool_response_truncate_side": mt.get("tool_response_truncate_side", "middle"),
            "tool_config_path": mt.get("tool_config_path"),
            "interaction_config_path": mt.get("interaction_config_path"),
            "format": mt.get("format", "hermes"),
            "use_inference_chat_template": mt.get("use_inference_chat_template", False),
            "tokenization_sanity_check_mode": mt.get("tokenization_sanity_check_mode", "strict"),
        }

    def _normalize_prompts(
        self, prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]]
    ) -> List[List[Message]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(prompts, dict):
            prompts = [prompts]

        normalized: List[List[Message]] = []
        for item in prompts:
            if isinstance(item, str):
                normalized.append([{"role": "user", "content": item}])
            else:
                normalized.append(item)
        return normalized

    def _build_dataproto(self, messages: List[List[Message]]) -> DataProto:
        # Use processor if it has apply_chat_template, otherwise fall back to tokenizer
        template_fn = getattr(self.processor, 'apply_chat_template', None)
        if template_fn is None:
            template_fn = getattr(self.tokenizer, 'apply_chat_template', None)
        if template_fn is None:
            raise ValueError("Neither processor nor tokenizer has apply_chat_template method")
        
        chat_texts = [
            template_fn(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in messages
        ]
        # Use the resolved prompt_length from rollout config for consistency
        prompt_len = getattr(self, '_prompt_length', self.config.get("max_prompt_length", 2048))
        tokenized = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_len,
        )

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = pad_sequence_to_length(tokenized["input_ids"], prompt_len, pad_token_id, left_pad=True)
        attention_mask = pad_sequence_to_length(tokenized["attention_mask"], prompt_len, 0, left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)
        position_ids = pad_sequence_to_length(position_ids, prompt_len, 0, left_pad=True)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )

        non_tensor = {
            "raw_prompt": np.array(messages, dtype=object),
            "tools_kwargs": np.array([{} for _ in messages], dtype=object),
        }

        data = DataProto(batch=batch, non_tensor_batch=non_tensor)
        data.meta_info.update(
            {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "do_sample": self.config.get("do_sample", True),
                "validate": False,
            }
        )
        return data

    def _sampling_kwargs(self, **overrides) -> tuple[Dict[str, Any], Optional[bool]]:
        # Use resolved response_length for consistency
        response_len = getattr(self, '_response_length', self.config.get("max_response_length", 512))
        params = {
            "temperature": overrides.get("temperature", self.config.get("temperature", 1.0)),
            "top_p": overrides.get("top_p", self.config.get("top_p", 1.0)),
            "top_k": overrides.get("top_k", self.config.get("top_k", -1)),
            "max_new_tokens": overrides.get("max_tokens", response_len),
        }
        do_sample = overrides.get("do_sample")
        if do_sample is False:
            params["temperature"] = 0.0
        return params, do_sample

    def generate(
        self,
        prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]],
        **kwargs,
    ) -> GenerationOutput:
        message_batches = self._normalize_prompts(prompts)
        data_proto = self._build_dataproto(message_batches)
        sampling_params, do_sample = self._sampling_kwargs(**kwargs)
        if do_sample is not None:
            data_proto.meta_info["do_sample"] = do_sample

        output = self.rollout.generate_sequences(data_proto, **sampling_params)
        return self._decode_output(output)

    def _decode_output(self, output: DataProto) -> GenerationOutput:
        output = output.to("cpu")
        batch = output.batch
        responses = batch["responses"]
        response_mask = batch.get("response_mask")
        completions: List[str] = []
        token_ids: List[List[int]] = []
        metadata: List[Dict[str, Any]] = []

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for idx in range(responses.shape[0]):
            resp = responses[idx]
            if response_mask is not None:
                mask = response_mask[idx].bool()
                llm_tokens = resp[mask]
            else:
                mask = resp != pad_token_id
                llm_tokens = resp[mask]

            ids = llm_tokens.tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            completions.append(text)
            token_ids.append(ids)
            metadata.append(
                {
                    "response_ids": resp.tolist(),
                    "response_mask": response_mask[idx].tolist() if response_mask is not None else None,
                }
            )

        return GenerationOutput(completions=completions, token_ids=token_ids, metadata=metadata)

    def shutdown(self):
        """Best-effort cleanup of the inference engine."""
        try:
            if hasattr(self.rollout, '_engine') and self.rollout._engine is not None:
                import asyncio
                loop = asyncio.get_event_loop()
                if hasattr(self.rollout._engine, 'shutdown'):
                    loop.run_until_complete(self.rollout._engine.shutdown())
        except Exception:
            pass  # Ignore cleanup errors
