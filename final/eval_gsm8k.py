#!/usr/bin/env python3
"""
GSM8K Evaluation using InferenceEngine.

Evaluates a model on the GSM8K math benchmark using the VERL SGLang inference engine.
Supports batched inference for optimal throughput.

Usage:
    python final/eval_gsm8k.py                           # Run with defaults
    python final/eval_gsm8k.py --batch-size 32           # Larger batches
    python final/eval_gsm8k.py --num-samples 100         # Quick test on 100 samples
    python final/eval_gsm8k.py --no-wandb                # Disable wandb logging
    python final/eval_gsm8k.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import yaml
from datasets import load_dataset
from tqdm import tqdm

from inf import InferenceEngine


@dataclass
class EvalResult:
    """Stores evaluation results for a single example."""
    question: str
    gold_answer: Optional[int]
    predicted_answer: Optional[int]
    completion: str
    correct: bool


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    correct: int = 0
    total: int = 0
    results: List[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def extract_answer(text: str) -> Optional[int]:
    """Extract final numeric answer from text.

    GSM8K gold answers have format: "... #### 42"
    Model responses might have the same or just end with a number.
    """
    # First try GSM8K format: #### <number>
    match = re.search(r'####\s*(-?\d[\d,]*)', text)
    if match:
        return int(match.group(1).replace(',', ''))

    # Fallback: find last number in text
    numbers = re.findall(r'-?\d[\d,]*', text)
    if numbers:
        return int(numbers[-1].replace(',', ''))

    return None


def format_prompt(question: str) -> str:
    """Format GSM8K question as a prompt."""
    return f"""Solve this math problem step by step. At the end, provide your final answer after "####".

Question: {question}

Solution:"""


def setup_wandb(config: dict, use_wandb: bool) -> Optional[object]:
    """Initialize wandb if enabled and available."""
    if not use_wandb:
        return None

    try:
        import wandb
        wandb_cfg = config.get("wandb", {})
        run = wandb.init(
            project=wandb_cfg.get("project", "gsm8k-eval"),
            name=wandb_cfg.get("run_name"),
            config=config,
            mode="online" if wandb_cfg.get("enabled", True) else "offline",
        )
        return run
    except ImportError:
        print("wandb not installed, skipping logging")
        return None
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return None


def log_wandb(run, metrics: dict):
    """Log metrics to wandb if available."""
    if run is not None:
        import wandb
        wandb.log(metrics)


def evaluate_batch(
    engine: InferenceEngine,
    prompts: List[str],
    gold_answers: List[Optional[int]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> List[EvalResult]:
    """Evaluate a batch of prompts and return results."""
    # Generate with greedy decoding for deterministic results
    gen_kwargs = {"temperature": temperature, "do_sample": False}
    if max_tokens is not None:
        gen_kwargs["max_tokens"] = max_tokens

    outputs = engine.generate(prompts, **gen_kwargs)

    results = []
    for prompt, gold, completion in zip(prompts, gold_answers, outputs.completions):
        pred = extract_answer(completion)
        correct = pred == gold and gold is not None

        results.append(EvalResult(
            question=prompt,
            gold_answer=gold,
            predicted_answer=pred,
            completion=completion,
            correct=correct,
        ))

    return results


def run_evaluation(
    engine: InferenceEngine,
    dataset,
    batch_size: int,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    wandb_run=None,
    verbose: bool = False,
) -> EvalMetrics:
    """Run full evaluation on dataset."""
    # Prepare all prompts and gold answers
    prompts = []
    gold_answers = []

    for item in dataset:
        prompts.append(format_prompt(item["question"]))
        gold_answers.append(extract_answer(item["answer"]))

    metrics = EvalMetrics()
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    print(f"\nRunning inference on {len(prompts)} samples in {num_batches} batches...")
    start_time = time.time()

    # Process in batches with progress bar
    with tqdm(total=len(prompts), desc="Evaluating", unit="samples") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_gold = gold_answers[i:i + batch_size]

            # Evaluate batch
            batch_results = evaluate_batch(
                engine, batch_prompts, batch_gold,
                temperature=temperature, max_tokens=max_tokens
            )

            # Update metrics
            for result in batch_results:
                metrics.results.append(result)
                metrics.total += 1
                if result.correct:
                    metrics.correct += 1

                # Log to wandb periodically
                log_wandb(wandb_run, {
                    "running_accuracy": metrics.accuracy,
                    "correct": metrics.correct,
                    "total": metrics.total,
                })

            # Print examples if verbose
            if verbose and metrics.total <= 3:
                for result in batch_results[:3 - (metrics.total - len(batch_results))]:
                    print(f"\n--- Example {metrics.total} ---")
                    print(f"Question: {result.question[:100]}...")
                    print(f"Response: {result.completion[:200]}...")
                    print(f"Predicted: {result.predicted_answer}, Gold: {result.gold_answer}, Correct: {result.correct}")

            pbar.update(len(batch_prompts))
            pbar.set_postfix({"acc": f"{metrics.accuracy:.1%}"})

    elapsed = time.time() - start_time
    samples_per_sec = metrics.total / elapsed if elapsed > 0 else 0
    print(f"\nInference completed in {elapsed:.1f}s ({samples_per_sec:.1f} samples/sec)")

    return metrics


def print_results(metrics: EvalMetrics):
    """Print final evaluation results."""
    print(f"\n{'='*50}")
    print(f"GSM8K Evaluation Results")
    print(f"{'='*50}")
    print(f"Correct: {metrics.correct}/{metrics.total}")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on GSM8K math benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size for inference (overrides config)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for generation (overrides config)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including examples",
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})

    # CLI args override config
    batch_size = args.batch_size or eval_cfg.get("batch_size", 8)
    num_samples = args.num_samples or eval_cfg.get("num_samples")

    print(f"Model: {config.get('model_path', 'unknown')}")
    print(f"Batch size: {batch_size}")
    print(f"Samples: {num_samples or 'all'}")
    print(f"Temperature: {args.temperature}")

    # Setup wandb
    wandb_run = setup_wandb(config, use_wandb=not args.no_wandb)

    engine = None
    try:
        # Load GSM8K test set
        print("\nLoading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"Evaluating on {len(dataset)} samples")

        # Initialize inference engine
        print("\nInitializing InferenceEngine...")
        engine = InferenceEngine(args.config)
        print("Engine ready!")

        # Run evaluation
        metrics = run_evaluation(
            engine=engine,
            dataset=dataset,
            batch_size=batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            wandb_run=wandb_run,
            verbose=args.verbose,
        )

        # Print results
        print_results(metrics)

        # Log final metrics to wandb
        log_wandb(wandb_run, {
            "final_accuracy": metrics.accuracy,
            "final_correct": metrics.correct,
            "final_total": metrics.total,
        })

        # Finish wandb
        if wandb_run is not None:
            import wandb
            wandb.finish()

        # Return success if we got results
        return 0 if metrics.total > 0 else 1

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 130

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if engine is not None:
            print("\nCleaning up...")
            engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
