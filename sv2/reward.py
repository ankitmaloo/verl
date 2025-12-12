"""
sv2/reward.py - Reward computation for multi-turn rollouts.

Provides:
1. GSM8K reward function (rule-based, extracts #### answer)
2. Simple reward manager that works with DataProto
3. Utility to compute rewards for a batch of rollouts
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from verl import DataProto


# =============================================================================
# GSM8K Reward Function (from verl/utils/reward_score/gsm8k.py)
# =============================================================================

_SOLUTION_CLIP_CHARS = 300


def extract_gsm8k_answer(solution_str: str, method: str = "strict") -> str | None:
    """
    Extract the final answer from a GSM8K solution string.

    Args:
        solution_str: The model's response
        method: "strict" (requires #### format) or "flexible" (last number)

    Returns:
        Extracted answer string or None if not found
    """
    assert method in ["strict", "flexible"]

    # Only look at the last 300 characters (answer is usually at the end)
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # Look for #### followed by a number
        solutions = re.findall(r"####\s*([\-]?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        # Take the last match, clean up commas and dollar signs
        return solutions[-1].replace(",", "").replace("$", "")

    elif method == "flexible":
        # Find all numbers, take the last valid one
        numbers = re.findall(r"([\-]?[0-9\.\,]+)", solution_str)
        if len(numbers) == 0:
            return None
        invalid = ["", "."]
        for answer in reversed(numbers):
            if answer not in invalid:
                return answer
        return None

    return None


def compute_gsm8k_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    correct_score: float = 1.0,
) -> float:
    """
    Compute GSM8K reward score.

    Args:
        solution_str: Model's response
        ground_truth: Expected answer (just the number)
        method: "strict" or "flexible" extraction
        format_score: Score if format is correct but answer wrong
        correct_score: Score if answer is correct

    Returns:
        Reward score (0.0, format_score, or correct_score)
    """
    answer = extract_gsm8k_answer(solution_str, method=method)
    if answer is None:
        return 0.0
    if answer == ground_truth:
        return correct_score
    return format_score


# =============================================================================
# Generic Reward Dispatcher
# =============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict:
    """
    Compute reward score based on data source.

    This is a simplified version of verl's default_compute_score.
    Add more data sources as needed.

    Args:
        data_source: Dataset identifier (e.g., "openai/gsm8k")
        solution_str: Model's response
        ground_truth: Expected answer
        extra_info: Additional info (unused for GSM8K)
        **kwargs: Additional arguments

    Returns:
        Score (float) or dict with score and metadata
    """
    if data_source == "openai/gsm8k" or data_source == "gsm8k":
        return compute_gsm8k_score(solution_str, ground_truth)
    else:
        # Default: try GSM8K-style extraction
        # This is a fallback - you may want to raise an error instead
        print(f"[sv2/reward] Warning: Unknown data_source={data_source!r}, using GSM8K scorer")
        return compute_gsm8k_score(solution_str, ground_truth)


# =============================================================================
# Reward Manager (simplified from NaiveRewardManager)
# =============================================================================

class Sv2RewardManager:
    """
    Simple reward manager for sv2 rollouts.

    Computes rewards for a batch of DataProto rollouts using the compute_score function.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        compute_score_fn=None,
        reward_fn_key: str = "data_source",
        num_examine: int = 1,
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding responses
            compute_score_fn: Custom scoring function (defaults to compute_score)
            reward_fn_key: Key in non_tensor_batch for data source
            num_examine: Number of samples to print for debugging
        """
        self.tokenizer = tokenizer
        self.compute_score_fn = compute_score_fn or compute_score
        self.reward_fn_key = reward_fn_key
        self.num_examine = num_examine

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of rollouts.

        Args:
            data: DataProto with prompts, responses, and reward_model info
            return_dict: If True, return dict with reward_tensor and extra_info

        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed = {}

        for i in range(len(data)):
            data_item = data[i]

            # Extract prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Get ground truth and data source
            reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model_info.get("ground_truth", "")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            # Compute score
            score = self.compute_score_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score.get("score", 0.0)
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = float(score)

            # Place reward at the last valid response token
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            # Debug printing
            if data_source not in already_printed:
                already_printed[data_source] = 0
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                print(f"[sv2/reward] === Sample {i} ===")
                print(f"[sv2/reward] data_source: {data_source}")
                print(f"[sv2/reward] prompt: {prompt_str[:200]}...")
                print(f"[sv2/reward] response: {response_str[:500]}...")
                print(f"[sv2/reward] ground_truth: {ground_truth}")
                print(f"[sv2/reward] score: {score}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor


def compute_batch_rewards(
    data: DataProto,
    tokenizer: PreTrainedTokenizer,
    num_examine: int = 1,
) -> tuple[torch.Tensor, dict[str, list]]:
    """
    Convenience function to compute rewards for a batch.

    Args:
        data: DataProto with rollout results
        tokenizer: Tokenizer for decoding
        num_examine: Number of samples to print

    Returns:
        Tuple of (reward_tensor, extra_info_dict)
    """
    manager = Sv2RewardManager(tokenizer=tokenizer, num_examine=num_examine)
    result = manager(data, return_dict=True)
    return result["reward_tensor"], result.get("reward_extra_info", {})
