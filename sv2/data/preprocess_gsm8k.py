"""
Preprocess GSM8K dataset for sv2 multi-turn rollouts with interaction support.

Creates parquet files with:
- prompt: Chat format messages
- extra_info.interaction_kwargs: For code verification interaction
- reward_model.ground_truth: For reward computation

Usage (run from sv2/ folder):
  python -m data.preprocess_gsm8k
  # Creates data/gsm8k/train.parquet and data/gsm8k/test.parquet
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import datasets


def extract_solution(solution_str: str) -> str:
    """Extract numeric answer from GSM8K solution string."""
    match = re.search(r"#### (-?[0-9.,]+)", solution_str)
    if match is None:
        return ""
    return match.group(1).replace(",", "")


def make_map_fn(split: str, interaction_name: str | None = None):
    """Create a mapping function for dataset processing."""

    def process_fn(example, idx):
        question_raw = example["question"]
        answer_raw = example["answer"]
        solution = extract_solution(answer_raw)

        instruction = "Let's think step by step and output the final answer after `####`."
        question = f"{question_raw} {instruction}"

        data = {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a math expert. You are given a question and you need to solve it step by step. "
                        "If asked to verify, write Python code to check your answer. "
                        "Put your final answer in the format of `#### <answer>`."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

        # Add interaction_kwargs if interaction_name is specified
        if interaction_name:
            data["extra_info"]["interaction_kwargs"] = {
                "name": interaction_name,
                "ground_truth": solution,
                "query": question,
            }

        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K for sv2")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/gsm8k",
        help="Output directory for parquet files (relative to cwd)",
    )
    parser.add_argument(
        "--interaction_name",
        type=str,
        default="code_verify",
        help="Interaction name to embed in extra_info.interaction_kwargs.name",
    )
    parser.add_argument(
        "--no_interaction",
        action="store_true",
        help="Don't add interaction_kwargs (use for basic rollouts)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per split (for testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_name = None if args.no_interaction else args.interaction_name

    print(f"Loading GSM8K dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if args.limit:
        train_dataset = train_dataset.select(range(min(args.limit, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
        print(f"Limited to {args.limit} samples per split")

    print(f"Processing train split ({len(train_dataset)} samples)...")
    train_dataset = train_dataset.map(
        make_map_fn("train", interaction_name),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )

    print(f"Processing test split ({len(test_dataset)} samples)...")
    test_dataset = test_dataset.map(
        make_map_fn("test", interaction_name),
        with_indices=True,
        remove_columns=test_dataset.column_names,
    )

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    print(f"\nOutput files:")
    print(f"  Train: {train_path} ({len(train_dataset)} samples)")
    print(f"  Test:  {test_path} ({len(test_dataset)} samples)")

    if interaction_name:
        print(f"\nInteraction: {interaction_name}")
        print("  Each sample has extra_info.interaction_kwargs.name set.")
    else:
        print("\nNo interaction_kwargs added.")
        print("  Use sv2.interaction_name=<name> at runtime to enable interactions.")

    print("\nUsage (from sv2/ folder):")
    print(f"  python -m main_ppo_multiturn_toolcall \\")
    print(f"    data.train_files={train_path} data.val_files={test_path} \\")
    if interaction_name:
        print(f"    actor_rollout_ref.rollout.multi_turn.interaction_config_path=config/interaction_config.yaml")
    else:
        print(f"    actor_rollout_ref.rollout.multi_turn.interaction_config_path=config/interaction_config.yaml \\")
        print(f"    sv2.interaction_name=code_verify")


if __name__ == "__main__":
    main()
