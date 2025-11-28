"""
Inference Engine for VERL Custom Training.

Provides both VLLM and SGLang backends with support for:
- Single-turn inference
- Multi-turn chat
- Batch processing
- Log probability tracking
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Container for generation outputs."""
    sequences: List[str]  # Generated text sequences
    tokens: np.ndarray  # Token IDs, shape (batch_size, seq_len)
    log_probs: Optional[np.ndarray] = None  # Log probabilities, shape (batch_size, seq_len)
    prompt_tokens: Optional[List[int]] = None
    completion_tokens: Optional[List[int]] = None


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model (HF model ID or local path)
            config: Configuration dictionary with inference parameters
        """
        self.model_path = model_path
        self.config = config
        self.tokenizer = None
        self.engine = None
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationOutput:
        """Generate responses for batch of prompts."""
        pass
    
    @abstractmethod
    def get_logits(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Get logits for prompts (for computing log probabilities)."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup resources."""
        pass


class VLLMInferenceEngine(BaseInferenceEngine):
    """VLLM-based inference engine."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize VLLM engine.
        
        Args:
            model_path: HF model ID or local path
            config: Configuration dict with keys:
                - device: cuda/cpu
                - dtype: bfloat16/float16/float32
                - tensor_parallel_size: TP degree
                - max_model_len: Max sequence length
                - trust_remote_code: Whether to trust remote code
        """
        super().__init__(model_path, config)
        
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("vllm not installed. Install with: pip install vllm")
        
        self.SamplingParams = SamplingParams
        
        # Initialize tokenizer
        trust_remote_code = config.get("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Set default tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize VLLM engine
        dtype = config.get("dtype", "bfloat16")
        tensor_parallel_size = config.get("tensor_parallel_size", 1)
        max_model_len = config.get("max_model_len", 2048)
        
        logger.info(f"Loading VLLM model: {model_path}")
        self.engine = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            disable_log_stats=True,
        )
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate responses using VLLM.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            GenerationOutput with sequences and metadata
        """
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Generate
        outputs = self.engine.generate(prompts, sampling_params)
        
        sequences = [output.outputs[0].text for output in outputs]
        
        # Tokenize outputs
        tokens_list = []
        for seq in sequences:
            token_ids = self.tokenizer.encode(seq)
            tokens_list.append(token_ids)
        
        # Pad to same length
        max_len = max(len(t) for t in tokens_list)
        tokens = np.zeros((len(tokens_list), max_len), dtype=np.int32)
        for i, t in enumerate(tokens_list):
            tokens[i, :len(t)] = t
        
        return GenerationOutput(
            sequences=sequences,
            tokens=tokens,
            log_probs=None,
        )
    
    def get_logits(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Get logits for prompts.
        
        Args:
            prompts: List of prompts
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            token_ids: Flattened token IDs
        """
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.engine.model(
                input_ids=input_ids.to(self.engine.device),
                attention_mask=attention_mask.to(self.engine.device) if attention_mask is not None else None,
                output_hidden_states=False,
            )
        
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        return logits, input_ids.flatten().tolist()
    
    def shutdown(self):
        """Cleanup VLLM engine."""
        if self.engine is not None:
            del self.engine
            torch.cuda.empty_cache()


class SGLangInferenceEngine(BaseInferenceEngine):
    """SGLang-based inference engine."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize SGLang engine.
        
        Args:
            model_path: HF model ID or local path
            config: Configuration dict
        """
        super().__init__(model_path, config)
        
        try:
            import sglang as sgl
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("sglang not installed. Install with: pip install sglang")
        
        self.sgl = sgl
        
        # Initialize tokenizer
        trust_remote_code = config.get("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize SGLang runtime
        logger.info(f"Loading SGLang model: {model_path}")
        self.engine = sgl.Runtime(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
        )
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate responses using SGLang.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            GenerationOutput with sequences
        """
        sequences = []
        
        for prompt in prompts:
            # SGLang generate
            output = self.engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            sequences.append(output)
        
        # Tokenize
        tokens_list = []
        for seq in sequences:
            token_ids = self.tokenizer.encode(seq)
            tokens_list.append(token_ids)
        
        max_len = max(len(t) for t in tokens_list)
        tokens = np.zeros((len(tokens_list), max_len), dtype=np.int32)
        for i, t in enumerate(tokens_list):
            tokens[i, :len(t)] = t
        
        return GenerationOutput(
            sequences=sequences,
            tokens=tokens,
        )
    
    def get_logits(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Get logits for prompts using SGLang.
        
        Args:
            prompts: List of prompts
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            token_ids: Flattened token IDs
        """
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        input_ids = encoded["input_ids"]
        
        # SGLang forward pass
        logits = self.engine.forward(input_ids.tolist())
        
        return logits, input_ids.flatten().tolist()
    
    def shutdown(self):
        """Cleanup SGLang engine."""
        if self.engine is not None:
            self.engine.shutdown()


class MultiTurnInferenceEngine:
    """
    Wrapper for multi-turn conversation with tool calling.
    
    Manages conversation history and integrates with environment and tools.
    """
    
    def __init__(
        self,
        engine: BaseInferenceEngine,
        max_turns: int = 5,
        system_prompt: Optional[str] = None,
        tool_manager: Optional[Any] = None,  # MultiTurnManager
        enable_tools: bool = False,
    ):
        """
        Initialize multi-turn engine.
        
        Args:
            engine: Underlying inference engine
            max_turns: Maximum conversation turns
            system_prompt: System prompt for chat format
            tool_manager: Optional MultiTurnManager for tool calling
            enable_tools: Whether to parse and execute tool calls
        """
        self.engine = engine
        self.max_turns = max_turns
        self.tool_manager = tool_manager
        self.enable_tools = enable_tools
        
        # If tools are enabled, use tool manager's system prompt
        if enable_tools and tool_manager:
            self.system_prompt = tool_manager.get_system_prompt()
        else:
            self.system_prompt = system_prompt or "You are a helpful assistant."
        
        self.conversation_history: List[Dict[str, str]] = []
    
    def start_conversation(self, initial_prompt: str):
        """Start a new conversation."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_prompt},
        ]
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def format_conversation(self) -> str:
        """Format conversation history as prompt text."""
        # TODO: Use proper chat template (ChatML, Llama2, etc.)
        prompt = ""
        for msg in self.conversation_history:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += "Assistant:"
        return prompt
    
    def step(
        self,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        environment_obs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate next response in conversation.
        
        Args:
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            environment_obs: Current observation from environment
            
        Returns:
            Dictionary with response, tool_call (if any), and metadata
        """
        prompt = self.format_conversation()
        
        output = self.engine.generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        response = output.sequences[0].strip()
        
        # Add to history
        self.add_message("assistant", response)
        
        result = {
            "response": response,
            "tool_call": None,
            "tool_result": None,
        }
        
        # If tools enabled, process tool calls
        if self.enable_tools and self.tool_manager:
            turn = self.tool_manager.step(response, environment_obs)
            result["tool_call"] = turn.tool_call
            result["tool_result"] = turn.observation
            
            # Add tool result to conversation
            if turn.observation:
                self.add_message("user", turn.observation)
        
        return result
    
    def run_episode(
        self,
        environment,
        initial_prompt: str,
        max_turns: Optional[int] = None,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Run a full multi-turn episode with environment and tools.
        
        Args:
            environment: Environment instance
            initial_prompt: Initial prompt to send
            max_turns: Max turns (overrides default)
            **gen_kwargs: Generation parameters
            
        Returns:
            Episode summary with turns, rewards, tool calls
        """
        max_turns = max_turns or self.max_turns
        
        # Reset tracking
        self.conversation_history = []
        if self.enable_tools and self.tool_manager:
            self.tool_manager.reset()
        
        # Reset environment
        obs = environment.reset()
        
        # Start conversation with environment state
        full_prompt = f"{initial_prompt}\n\n{obs}"
        self.start_conversation(full_prompt)
        
        turns_data = []
        total_reward = 0.0
        
        for turn_idx in range(max_turns):
            # Generate response (may include tool calls)
            step_result = self.step(environment_obs=obs, **gen_kwargs)
            response = step_result["response"]
            tool_call = step_result["tool_call"]
            
            turns_data.append({
                "turn": turn_idx + 1,
                "response": response,
                "tool_call": tool_call,
                "observation": obs,
            })
            
            # Execute in environment if tool not used
            # (if tool was used, tool_result already contains the observation)
            if tool_call is None:
                # Parse response as action if no tool
                try:
                    obs, reward, done, info = environment.step(response)
                    total_reward += reward
                except Exception as e:
                    logger.warning(f"Environment step failed: {e}")
                    obs = f"Error: {str(e)}"
                    done = True
                
                # Add observation to conversation
                self.add_message("user", obs)
            else:
                # Tool was executed, use tool result as next observation
                obs = step_result["tool_result"]
                # Tool reward is captured in tool execution
            
            if done or (self.enable_tools and self.tool_manager and not self.tool_manager.should_continue()):
                break
        
        # Generate episode summary
        summary = {
            "num_turns": len(turns_data),
            "total_reward": total_reward,
            "turns": turns_data,
            "conversation_length": len(self.conversation_history),
        }
        
        # Add tool manager summary if available
        if self.enable_tools and self.tool_manager:
            summary.update({"tools": self.tool_manager.get_episode_summary()})
        
        return summary


def create_inference_engine(
    model_path: str,
    backend: str = "vllm",
    config: Optional[Dict[str, Any]] = None,
) -> BaseInferenceEngine:
    """
    Factory function to create inference engine.
    
    Args:
        model_path: Model path or HF model ID
        backend: "vllm" or "sglang"
        config: Configuration dictionary
        
    Returns:
        Inference engine instance
    """
    if config is None:
        config = {}
    
    if backend == "vllm":
        return VLLMInferenceEngine(model_path, config)
    elif backend == "sglang":
        return SGLangInferenceEngine(model_path, config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# BATCH INFERENCE HELPER
# ============================================================================

class BatchInferenceManager:
    """Manages batched inference with VERL."""
    
    def __init__(
        self,
        engine: BaseInferenceEngine,
        batch_size: int = 32,
    ):
        """
        Initialize batch manager.
        
        Args:
            engine: Inference engine
            batch_size: Batch size for inference
        """
        self.engine = engine
        self.batch_size = batch_size
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[GenerationOutput]:
        """
        Generate responses for large batch of prompts.
        
        Args:
            prompts: List of prompts
            max_tokens: Max tokens per response
            temperature: Sampling temperature
            top_p: Nucleus sampling
            
        Returns:
            List of GenerationOutput objects
        """
        outputs = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            
            output = self.engine.generate(
                batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            outputs.append(output)
        
        # Merge outputs
        all_sequences = []
        all_tokens = []
        
        for output in outputs:
            all_sequences.extend(output.sequences)
            all_tokens.append(output.tokens)
        
        # Concatenate token arrays with padding
        max_len = max(t.shape[1] for t in all_tokens)
        merged_tokens = np.zeros(
            (sum(t.shape[0] for t in all_tokens), max_len),
            dtype=np.int32
        )
        
        idx = 0
        for tokens in all_tokens:
            n = tokens.shape[0]
            merged_tokens[idx:idx+n, :tokens.shape[1]] = tokens
            idx += n
        
        return GenerationOutput(
            sequences=all_sequences,
            tokens=merged_tokens,
        )
