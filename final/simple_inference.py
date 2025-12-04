"""
Direct SGLang inference - NO VERL overhead.
Works in Colab notebooks without crashing.

VERL is for distributed RL training, not simple inference!
"""

import yaml
import sglang as sgl
from typing import List, Union
from transformers import AutoTokenizer


class SimpleInference:
    """
    Direct SGLang inference - lightweight, Colab-friendly.
    No VERL, no torch.distributed, no crashes!
    """
    
    def __init__(self, config_path="config.yaml"):
        """Initialize from config."""
        self.config = self._load_config(config_path)
        self.engine = None
        self.tokenizer = None
        self._init()
    
    def _load_config(self, config_path):
        """Load config with defaults."""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Config {config_path} not found, using defaults")
            config = {}
        
        # Colab-friendly defaults
        defaults = {
            "model_path": "Qwen/Qwen3-0.6B",
            "trust_remote_code": False,
            "num_gpus": 1,
            "gpu_memory_utilization": 0.4,  # Conservative for Colab T4
            "dtype": "auto",
            "max_prompt_length": 1024,  # Reduced for T4
            "max_response_length": 256,  # Reduced for T4
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "do_sample": True,
        }
        
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
        
        return config
    
    def _init(self):
        """Initialize SGLang engine directly."""
        print(f"Initializing SGLang (direct, no VERL overhead)...")
        print(f"  Model: {self.config['model_path']}")
        print(f"  Memory: {self.config['gpu_memory_utilization']}")
        
        # Direct SGLang initialization
        self.engine = sgl.Engine(
            model_path=self.config['model_path'],
            tp_size=self.config['num_gpus'],
            dtype=self.config['dtype'],
            trust_remote_code=self.config['trust_remote_code'],
            mem_fraction_static=self.config['gpu_memory_utilization'],
            context_length=self.config['max_prompt_length'] + self.config['max_response_length'],
            
            # CRITICAL: T4-compatible settings
            attention_backend="flashinfer",  # Works on T4
            disable_cuda_graph=True,  # Reduce memory
            disable_flashinfer=False,  # Use flashinfer
            
            # Logging
            log_level="error",
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_path'],
            trust_remote_code=self.config['trust_remote_code'],
            padding_side='left',
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("✓ Initialized (lightweight, Colab-safe)")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        apply_chat_template: bool = True,
        **sampling_overrides
    ) -> List[str]:
        """
        Generate responses.
        
        Args:
            prompts: Single string or list
            apply_chat_template: Whether to use chat template
            **sampling_overrides: temperature, top_p, etc.
        
        Returns:
            List of response strings
        """
        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Format with chat template
        if apply_chat_template:
            formatted = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                formatted.append(text)
            prompts = formatted
        
        # Build sampling params
        sampling_params = {
            "temperature": sampling_overrides.get("temperature", self.config["temperature"]),
            "top_p": sampling_overrides.get("top_p", self.config["top_p"]),
            "top_k": sampling_overrides.get("top_k", self.config["top_k"]),
            "max_new_tokens": sampling_overrides.get("max_new_tokens", self.config["max_response_length"]),
        }
        
        do_sample = sampling_overrides.get("do_sample", self.config["do_sample"])
        if not do_sample:
            sampling_params["temperature"] = 0.0
        
        # Generate
        print(f"Generating {len(prompts)} responses...")
        outputs = self.engine.generate(
            prompt=prompts,
            sampling_params=sampling_params,
        )
        
        # Extract text
        responses = []
        for output in outputs:
            text = output.get("text", "")
            responses.append(text)
        
        return responses
    
    def shutdown(self):
        """Cleanup."""
        if self.engine:
            self.engine.shutdown()
            print("✓ Engine shutdown")


def main():
    """Example usage."""
    # Initialize
    engine = SimpleInference("config.yaml")
    
    # Example
    print("\n" + "="*80)
    print("Testing inference (no VERL overhead)")
    print("="*80)
    
    responses = engine.generate([
        "What is 2+2?",
        "Write a haiku about clouds.",
    ])
    
    for i, response in enumerate(responses, 1):
        print(f"\n{i}. {response}")
    
    # Cleanup
    print("\n" + "="*80)
    engine.shutdown()
    print("✓ Complete!")


if __name__ == "__main__":
    main()
