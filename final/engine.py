"""Unified VERL Inference Engine supporting both SGLang and vLLM backends.

Provides a simple, consistent API for generating rollouts with either backend.
Switch backends via config: rollout.name: "sglang" or "vllm"

Usage:
    from final.engine import InferenceEngine

    engine = InferenceEngine("final/config.yaml")
    output = engine.generate(["What is 2+2?"])
    print(output.completions[0])
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.config import HFModelConfig

Message = Dict[str, Any]


@dataclass
class GenerationOutput:
    """Container for generation outputs from both backends."""
    completions: List[str]
    token_ids: List[List[int]]
    log_probs: Optional[List[List[float]]] = None
    metadata: Optional[List[Dict[str, Any]]] = None


def _ensure_dist_initialized():
    """Initialize minimal distributed environment for single-GPU inference."""
    import torch.distributed as dist
    from verl.utils.device import get_visible_devices_keyword

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


class BaseEngine(ABC):
    """Abstract base class with shared functionality for inference engines."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = HFModelConfig(
            path=config["model_path"],
            trust_remote_code=config.get("trust_remote_code", False),
        )
        self.tokenizer = self.model_config.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.processor = self.model_config.get_processor()

        # Get rollout config
        rollout_cfg = config.get("rollout", {}) or config
        self._prompt_length = rollout_cfg.get("prompt_length", config.get("max_prompt_length", 2048))
        self._response_length = rollout_cfg.get("response_length", config.get("max_response_length", 512))

    def _normalize_prompts(
        self, prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]]
    ) -> List[List[Message]]:
        """Normalize various prompt formats to List[List[Message]]."""
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

    def _apply_chat_template(self, messages: List[List[Message]]) -> List[str]:
        """Apply chat template to messages."""
        template_fn = getattr(self.processor, 'apply_chat_template', None)
        if template_fn is None:
            template_fn = getattr(self.tokenizer, 'apply_chat_template', None)
        if template_fn is None:
            raise ValueError("Neither processor nor tokenizer has apply_chat_template method")

        return [
            template_fn(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in messages
        ]

    def _tokenize_prompts(self, chat_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompts with left padding."""
        tokenized = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._prompt_length,
        )

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = pad_sequence_to_length(
            tokenized["input_ids"], self._prompt_length, pad_token_id, left_pad=True
        )
        attention_mask = pad_sequence_to_length(
            tokenized["attention_mask"], self._prompt_length, 0, left_pad=True
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        position_ids = pad_sequence_to_length(position_ids, self._prompt_length, 0, left_pad=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pad_token_id": pad_token_id,
        }

    def _build_dataproto(self, messages: List[List[Message]]) -> DataProto:
        """Build DataProto from messages (for SGLang backend)."""
        chat_texts = self._apply_chat_template(messages)
        tokenized = self._tokenize_prompts(chat_texts)

        batch = TensorDict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "position_ids": tokenized["position_ids"],
            },
            batch_size=tokenized["input_ids"].shape[0],
        )

        non_tensor = {
            "raw_prompt": np.array(messages, dtype=object),
            "tools_kwargs": np.array([{} for _ in messages], dtype=object),
        }

        data = DataProto(batch=batch, non_tensor_batch=non_tensor)
        data.meta_info.update({
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": tokenized["pad_token_id"],
            "do_sample": self.config.get("do_sample", True),
            "validate": False,
        })
        return data

    def _get_sampling_params(self, **overrides) -> Dict[str, Any]:
        """Get sampling parameters from config with overrides."""
        rollout_cfg = self.config.get("rollout", {}) or self.config
        params = {
            "temperature": overrides.get("temperature", rollout_cfg.get("temperature", 1.0)),
            "top_p": overrides.get("top_p", rollout_cfg.get("top_p", 1.0)),
            "top_k": overrides.get("top_k", rollout_cfg.get("top_k", -1)),
            "max_new_tokens": overrides.get("max_tokens", self._response_length),
        }
        if overrides.get("do_sample") is False:
            params["temperature"] = 0.0
        return params

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]],
        **kwargs,
    ) -> GenerationOutput:
        """Generate completions for prompts."""
        pass

    @abstractmethod
    def shutdown(self):
        """Cleanup resources."""
        pass


class SGLangEngine(BaseEngine):
    """SGLang backend using VERL's SGLangRollout."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        _ensure_dist_initialized()

        # Import here to avoid import errors when sglang not installed
        from omegaconf import OmegaConf
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import RolloutConfig
        from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

        os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "True")

        rollout_conf = self._build_rollout_config(OmegaConf, omega_conf_to_dataclass, RolloutConfig)
        self.rollout = SGLangRollout(
            config=rollout_conf,
            model_config=self.model_config,
            device_mesh=None,
        )

    def _build_rollout_config(self, OmegaConf, omega_conf_to_dataclass, RolloutConfig):
        """Build RolloutConfig from config dict."""
        rollout_cfg = self.config.get("rollout", {}) or self.config

        rollout_dict = {
            "name": rollout_cfg.get("name", "sglang"),
            "mode": rollout_cfg.get("mode", "async"),
            "skip_tokenizer_init": rollout_cfg.get("skip_tokenizer_init", True),
            "prompt_length": self._prompt_length,
            "response_length": self._response_length,
            "max_model_len": rollout_cfg.get("max_model_len"),
            "max_num_seqs": rollout_cfg.get("max_num_seqs", 1024),
            "max_num_batched_tokens": rollout_cfg.get("max_num_batched_tokens", 8192),
            "temperature": rollout_cfg.get("temperature", 1.0),
            "top_p": rollout_cfg.get("top_p", 1.0),
            "top_k": rollout_cfg.get("top_k", -1),
            "do_sample": rollout_cfg.get("do_sample", True),
            "n": rollout_cfg.get("n", 1),
            "ignore_eos": rollout_cfg.get("ignore_eos", False),
            "val_kwargs": rollout_cfg.get("val_kwargs", {
                "temperature": 0, "top_k": -1, "top_p": 1.0, "do_sample": False, "n": 1,
            }),
            "tensor_model_parallel_size": rollout_cfg.get(
                "tensor_model_parallel_size", self.config.get("num_gpus", 1)
            ),
            "data_parallel_size": rollout_cfg.get("data_parallel_size", 1),
            "expert_parallel_size": rollout_cfg.get("expert_parallel_size", 1),
            "pipeline_model_parallel_size": rollout_cfg.get("pipeline_model_parallel_size", 1),
            "gpu_memory_utilization": rollout_cfg.get(
                "gpu_memory_utilization", self.config.get("gpu_memory_utilization", 0.5)
            ),
            "dtype": rollout_cfg.get("dtype", self.config.get("dtype", "bfloat16")),
            "free_cache_engine": rollout_cfg.get("free_cache_engine", True),
            "enforce_eager": rollout_cfg.get("enforce_eager", False),
            "cudagraph_capture_sizes": rollout_cfg.get("cudagraph_capture_sizes"),
            "enable_chunked_prefill": rollout_cfg.get("enable_chunked_prefill", True),
            "enable_prefix_caching": rollout_cfg.get("enable_prefix_caching", True),
            "load_format": rollout_cfg.get("load_format", "auto"),
            "engine_kwargs": rollout_cfg.get("engine_kwargs", {}),
            "calculate_log_probs": rollout_cfg.get(
                "calculate_log_probs", self.config.get("return_logprobs", False)
            ),
            "disable_log_stats": rollout_cfg.get("disable_log_stats", True),
            "over_sample_rate": rollout_cfg.get("over_sample_rate", 0.0),
            "update_weights_bucket_megabytes": rollout_cfg.get("update_weights_bucket_megabytes", 512),
            "multi_stage_wake_up": rollout_cfg.get("multi_stage_wake_up", False),
            "multi_turn": self._build_multi_turn_cfg(rollout_cfg),
            "skip_rollout": rollout_cfg.get("skip_rollout", False),
            "skip_dump_dir": rollout_cfg.get("skip_dump_dir", "/tmp/rollout_dump"),
            "profiler": rollout_cfg.get("profiler"),
        }

        omega = OmegaConf.create(rollout_dict)
        return omega_conf_to_dataclass(omega, dataclass_type=RolloutConfig)

    def _build_multi_turn_cfg(self, rollout_cfg: Dict) -> Dict[str, Any]:
        """Build multi-turn config."""
        mt = rollout_cfg.get("multi_turn", {}) or {}
        return {
            "enable": mt.get("enable", False),
            "max_assistant_turns": mt.get("max_assistant_turns", mt.get("max_turns")),
            "max_user_turns": mt.get("max_user_turns", mt.get("max_turns")),
            "max_parallel_calls": mt.get("max_parallel_calls", 1),
            "max_tool_response_length": mt.get("max_tool_response_length", 512),
            "tool_response_truncate_side": mt.get("tool_response_truncate_side", "middle"),
            "tool_config_path": mt.get("tool_config_path"),
            "interaction_config_path": mt.get("interaction_config_path"),
            "format": mt.get("format", "hermes"),
            "use_inference_chat_template": mt.get("use_inference_chat_template", False),
            "tokenization_sanity_check_mode": mt.get("tokenization_sanity_check_mode", "strict"),
        }

    def generate(
        self,
        prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]],
        **kwargs,
    ) -> GenerationOutput:
        message_batches = self._normalize_prompts(prompts)
        data_proto = self._build_dataproto(message_batches)
        sampling_params = self._get_sampling_params(**kwargs)

        do_sample = kwargs.get("do_sample")
        if do_sample is not None:
            data_proto.meta_info["do_sample"] = do_sample

        output = self.rollout.generate_sequences(data_proto, **sampling_params)
        return self._decode_output(output)

    def _decode_output(self, output: DataProto) -> GenerationOutput:
        """Decode SGLang output to GenerationOutput."""
        output = output.to("cpu")
        batch = output.batch
        responses = batch["responses"]
        response_mask = batch.get("response_mask")

        completions: List[str] = []
        token_ids: List[List[int]] = []
        log_probs: Optional[List[List[float]]] = None
        metadata: List[Dict[str, Any]] = []

        # Check for log probs
        if "log_probs" in batch:
            log_probs = []

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

            if log_probs is not None and "log_probs" in batch:
                lp = batch["log_probs"][idx]
                if response_mask is not None:
                    lp = lp[response_mask[idx].bool()]
                log_probs.append(lp.tolist())

            metadata.append({
                "response_ids": resp.tolist(),
                "response_mask": response_mask[idx].tolist() if response_mask is not None else None,
            })

        return GenerationOutput(
            completions=completions,
            token_ids=token_ids,
            log_probs=log_probs,
            metadata=metadata,
        )

    def shutdown(self):
        """Cleanup SGLang engine."""
        try:
            if hasattr(self.rollout, '_engine') and self.rollout._engine is not None:
                import asyncio
                loop = asyncio.get_event_loop()
                if hasattr(self.rollout._engine, 'shutdown'):
                    loop.run_until_complete(self.rollout._engine.shutdown())
        except Exception:
            pass


class VLLMEngine(BaseEngine):
    """vLLM backend using vllm.LLM directly (no Ray)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm not installed. Install with: pip install vllm")

        self.SamplingParams = SamplingParams

        rollout_cfg = config.get("rollout", {}) or config

        # Build vLLM engine
        max_model_len = rollout_cfg.get("max_model_len")
        if max_model_len is None:
            max_model_len = self._prompt_length + self._response_length

        engine_kwargs = rollout_cfg.get("engine_kwargs", {}).get("vllm", {}) or {}

        self.engine = LLM(
            model=config["model_path"],
            dtype=rollout_cfg.get("dtype", "auto"),
            tensor_parallel_size=rollout_cfg.get("tensor_model_parallel_size", 1),
            max_model_len=max_model_len,
            gpu_memory_utilization=rollout_cfg.get("gpu_memory_utilization", 0.5),
            trust_remote_code=config.get("trust_remote_code", False),
            enforce_eager=rollout_cfg.get("enforce_eager", False),
            enable_chunked_prefill=rollout_cfg.get("enable_chunked_prefill", True),
            enable_prefix_caching=rollout_cfg.get("enable_prefix_caching", True),
            disable_log_stats=rollout_cfg.get("disable_log_stats", True),
            **engine_kwargs,
        )

        self._return_logprobs = rollout_cfg.get(
            "calculate_log_probs", config.get("return_logprobs", False)
        )

    def generate(
        self,
        prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]],
        **kwargs,
    ) -> GenerationOutput:
        message_batches = self._normalize_prompts(prompts)
        chat_texts = self._apply_chat_template(message_batches)

        # Build sampling params
        params = self._get_sampling_params(**kwargs)
        return_logprobs = kwargs.get("return_logprobs", self._return_logprobs)

        sampling_params = self.SamplingParams(
            max_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"] if params["top_k"] > 0 else -1,
            logprobs=1 if return_logprobs else None,  # 1 = return top-1 logprob
        )

        # Generate
        outputs = self.engine.generate(chat_texts, sampling_params)

        return self._decode_output(outputs, return_logprobs)

    def _decode_output(self, outputs, return_logprobs: bool) -> GenerationOutput:
        """Decode vLLM outputs to GenerationOutput."""
        completions: List[str] = []
        token_ids: List[List[int]] = []
        log_probs: Optional[List[List[float]]] = [] if return_logprobs else None
        metadata: List[Dict[str, Any]] = []

        for output in outputs:
            # Get first completion (n=1)
            completion_output = output.outputs[0]
            text = completion_output.text
            ids = list(completion_output.token_ids)

            completions.append(text)
            token_ids.append(ids)

            if return_logprobs and completion_output.logprobs is not None:
                # Extract log prob for each generated token
                lps = []
                for i, logprobs_dict in enumerate(completion_output.logprobs):
                    if logprobs_dict and ids[i] in logprobs_dict:
                        lps.append(logprobs_dict[ids[i]].logprob)
                    else:
                        lps.append(0.0)  # fallback
                log_probs.append(lps)

            metadata.append({
                "prompt_token_ids": list(output.prompt_token_ids) if output.prompt_token_ids else None,
                "finish_reason": completion_output.finish_reason,
            })

        return GenerationOutput(
            completions=completions,
            token_ids=token_ids,
            log_probs=log_probs,
            metadata=metadata,
        )

    def shutdown(self):
        """Cleanup vLLM engine."""
        if self.engine is not None:
            del self.engine
            torch.cuda.empty_cache()


class InferenceEngine:
    """Unified inference engine that delegates to SGLang or vLLM backend.

    Backend is selected via rollout.name in config:
        - "sglang": Uses VERL's SGLangRollout
        - "vllm": Uses vllm.LLM directly

    Example:
        engine = InferenceEngine("config.yaml")
        output = engine.generate(["What is 2+2?"])
        print(output.completions[0])
    """

    def __init__(self, config_path: str):
        """Initialize engine from config file.

        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Determine backend from config
        rollout_cfg = self.config.get("rollout", {}) or {}
        backend_name = rollout_cfg.get("name", "sglang").lower()

        if backend_name == "sglang":
            self._engine = SGLangEngine(self.config)
        elif backend_name == "vllm":
            self._engine = VLLMEngine(self.config)
        else:
            raise ValueError(f"Unknown backend: {backend_name}. Use 'sglang' or 'vllm'")

        self.backend = backend_name

    @property
    def tokenizer(self):
        """Access the tokenizer."""
        return self._engine.tokenizer

    def generate(
        self,
        prompts: Union[str, Dict[str, Any], Sequence[Union[str, List[Message]]]],
        **kwargs,
    ) -> GenerationOutput:
        """Generate completions for prompts.

        Args:
            prompts: Input prompts. Can be:
                - Single string: "What is 2+2?"
                - List of strings: ["What is 2+2?", "Hello"]
                - List of message dicts: [[{"role": "user", "content": "Hi"}]]
            **kwargs: Override sampling params:
                - temperature: float
                - top_p: float
                - top_k: int
                - max_tokens: int
                - do_sample: bool
                - return_logprobs: bool (vLLM only, SGLang uses config)

        Returns:
            GenerationOutput with completions, token_ids, log_probs, metadata
        """
        return self._engine.generate(prompts, **kwargs)

    def shutdown(self):
        """Cleanup resources."""
        self._engine.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


# Convenience function
def create_engine(config_path: str) -> InferenceEngine:
    """Create an inference engine from config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        InferenceEngine instance
    """
    return InferenceEngine(config_path)
