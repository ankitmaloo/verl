"""
Simple inference wrapper using VERL's SGLang HTTP server engine.

This module provides a clean interface for inference with weight sync capability,
using VERL's AsyncHttpServerAdapter which launches an SGLang HTTP server.

Usage:
    from final.inference import InferenceEngine

    engine = InferenceEngine("final/config.yaml")
    outputs = engine.generate(["What is 2+2?"])
    print(outputs.completions)
"""

import yaml
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

from verl.workers.config import HFModelConfig


@dataclass
class GenerationOutput:
    """Output from generation."""
    completions: List[str]
    token_ids: List[List[int]]
    log_probs: Optional[List] = None


class InferenceEngine:
    """
    Inference engine using VERL's SGLang HTTP server adapter.

    Supports weight synchronization for RL training integration.
    Handles tokenization, chat templates internally.

    Args:
        config_path: Path to YAML config file

    Example:
        engine = InferenceEngine("config.yaml")

        # String prompts
        outputs = engine.generate(["What is 2+2?", "Hello"])

        # Chat messages
        outputs = engine.generate([[
            {"role": "user", "content": "Hello"}
        ]])

        # Override sampling
        outputs = engine.generate(["Be creative"], temperature=1.5)
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._init_model_config()
        self._init_server()

    def _init_model_config(self):
        """Initialize HFModelConfig from VERL for tokenizer/processor."""
        self.model_config = HFModelConfig(
            path=self.config["model_path"],
            trust_remote_code=self.config.get("trust_remote_code", False),
        )
        self.tokenizer = self.model_config.tokenizer
        self.tokenizer.padding_side = "left"
        self.processor = self.model_config.get_processor()

    def _init_server(self):
        """Initialize SGLang Engine directly (simpler than HTTP server)."""
        import sglang as sgl

        num_gpus = self.config.get("num_gpus", 1)

        self._engine = sgl.Engine(
            model_path=self.config["model_path"],
            tp_size=num_gpus,
            dtype=self.config.get("dtype", "bfloat16"),
            trust_remote_code=self.config.get("trust_remote_code", False),
            mem_fraction_static=self.config.get("gpu_memory_utilization", 0.5),
            context_length=self.config.get("max_prompt_length", 2048) + self.config.get("max_response_length", 512),
            disable_cuda_graph=True,
            log_level="info",
        )

    def generate(
        self,
        prompts: Union[str, List[str], List[List[Dict[str, str]]]],
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate completions for prompts.

        Args:
            prompts: Can be:
                - Single string: "What is 2+2?"
                - List of strings: ["What is 2+2?", "Hello"]
                - List of chat messages: [[{"role": "user", "content": "Hi"}]]
            **kwargs: Override sampling params (temperature, top_p, top_k, max_tokens)

        Returns:
            GenerationOutput with completions, token_ids, and optionally log_probs
        """
        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("prompts cannot be empty")

        # Build sampling params
        sampling_params = {
            "temperature": kwargs.get("temperature", self.config.get("temperature", 1.0)),
            "top_p": kwargs.get("top_p", self.config.get("top_p", 1.0)),
            "top_k": kwargs.get("top_k", self.config.get("top_k", -1)),
            "max_new_tokens": kwargs.get("max_tokens", self.config.get("max_response_length", 512)),
        }

        # Handle do_sample -> temperature
        if kwargs.get("do_sample") is False or (
            "do_sample" not in kwargs and not self.config.get("do_sample", True)
        ):
            sampling_params["temperature"] = 0.0

        # Apply chat template to all prompts
        texts = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            texts.append(text)

        # Generate using SGLang Engine (synchronous batch call)
        outputs = self._engine.generate(
            prompt=texts,
            sampling_params=sampling_params,
        )

        # Extract results
        completions = []
        token_ids_list = []

        for output in outputs:
            # Get output text and token ids
            output_text = output.get("text", "")
            output_ids = output.get("meta_info", {}).get("output_token_ids", [])

            if output_ids:
                token_ids_list.append(output_ids)
                # Decode ourselves to preserve special tokens
                completions.append(self.tokenizer.decode(output_ids, skip_special_tokens=False))
            else:
                completions.append(output_text)
                token_ids_list.append(self.tokenizer.encode(output_text, add_special_tokens=False))

        return GenerationOutput(
            completions=completions,
            token_ids=token_ids_list,
            log_probs=None,
        )

    def shutdown(self):
        """Shutdown the server and cleanup."""
        self._engine.shutdown()

    # ---- Tokenizer utilities ----

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        return self.tokenizer.encode(text)

    def detokenize(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """Apply the model's chat template to messages."""
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id
