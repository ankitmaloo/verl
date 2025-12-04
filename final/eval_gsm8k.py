"""
GSM8K Evaluation using InferenceEngine.

Tests the inference.py wrapper by running GSM8K math evaluation.

Usage:
    python final/eval_gsm8k.py
"""

import re
import yaml
from tqdm import tqdm
from datasets import load_dataset
import wandb

from inf import InferenceEngine

CONFIG_PATH = "final/config.yaml"


def extract_answer(text: str) -> int | None:
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


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})
    wandb_cfg = config.get("wandb", {})

    num_samples = eval_cfg.get("num_samples")
    batch_size = eval_cfg.get("batch_size", 1)

    # Initialize wandb (runs offline if no API key)
    wandb.init(
        project=wandb_cfg.get("project", "gsm8k-eval"),
        name=wandb_cfg.get("run_name"),
        config=config,
    )

    # Load GSM8K test set
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples")

    # Initialize inference engine
    print("Initializing InferenceEngine...")
    engine = InferenceEngine(CONFIG_PATH)

    # Evaluation loop
    correct = 0
    total = 0

    # Process in batches
    prompts = []
    gold_answers = []

    for item in dataset:
        prompts.append(format_prompt(item["question"]))
        gold_answers.append(extract_answer(item["answer"]))

    print("Running inference...")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_gold = gold_answers[i:i + batch_size]

        # Generate with greedy decoding for deterministic results
        outputs = engine.generate(batch_prompts, temperature=0.0, do_sample=False)

        for j, (completion, gold) in enumerate(zip(outputs.completions, batch_gold)):
            pred = extract_answer(completion)

            if pred == gold:
                correct += 1
            total += 1

            wandb.log({
                "running_accuracy": correct / total,
                "correct": correct,
                "total": total,
            })

            # Print first few examples
            if total <= 3:
                print(f"\n--- Example {total} ---")
                print(f"Question: {batch_prompts[j][:100]}...")
                print(f"Response: {completion[:200]}...")
                print(f"Predicted: {pred}, Gold: {gold}, Correct: {pred == gold}")

    # Final results
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"GSM8K Results")
    print(f"{'='*50}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")

    wandb.log({
        "final_accuracy": accuracy,
        "final_correct": correct,
        "final_total": total,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
