#!/usr/bin/env python3
"""
Create a custom reward function template.

Usage:
    python scripts/create_reward_template.py --task math --output my_reward.py
    python scripts/create_reward_template.py --task code --output my_reward.py
"""

import argparse
import sys


TEMPLATES = {
    "math": '''import torch
from verl import DataProto
import re

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    """Score math problems by comparing numeric answers."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.get('ground_truths', [None])[i]

        if ground_truth is None:
            continue

        response_ans = extract_answer(response)
        truth_ans = extract_answer(str(ground_truth))

        if response_ans is not None and truth_ans is not None:
            if abs(response_ans - truth_ans) < 1e-6:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        else:
            rewards[i] = 0.0

    return rewards

def extract_answer(text: str):
    """Extract numeric answer from text."""
    patterns = [
        r'answer\\s*(?:is|:)\\s*([-+]?\\d+\\.?\\d*)',
        r'(?:=|≈)\\s*([-+]?\\d+\\.?\\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass

    numbers = re.findall(r'[-+]?\\d+\\.?\\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass

    return None
''',

    "code": '''import torch
import subprocess
import tempfile
from verl import DataProto

def compute_reward(data: DataProto, test_timeout=5, **kwargs) -> torch.Tensor:
    """Score code by testing against test cases."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    test_cases = data.get('test_cases', [None] * batch_size)

    for i in range(batch_size):
        code = data.responses[i]
        tests = test_cases[i]

        if tests is None:
            continue

        passed_ratio = test_code(code, tests, test_timeout)
        rewards[i] = passed_ratio

    return rewards

def test_code(code: str, test_cases, timeout: int) -> float:
    """Execute code and test against test cases."""
    passed = 0
    total = len(test_cases)

    for test in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name

            result = subprocess.run(
                ['python', code_file],
                input=test['input'],
                capture_output=True,
                timeout=timeout,
                text=True,
            )

            if result.stdout.strip() == test['output'].strip():
                passed += 1

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    return passed / total if total > 0 else 0.0
''',

    "qa": '''import torch
from difflib import SequenceMatcher
from verl import DataProto

def compute_reward(data: DataProto, use_partial=True, **kwargs) -> torch.Tensor:
    """Score QA by comparing with ground truth."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.get('ground_truths', [None])[i]

        if ground_truth is None:
            continue

        if use_partial:
            similarity = SequenceMatcher(None, response, ground_truth).ratio()
            rewards[i] = similarity
        else:
            rewards[i] = 1.0 if response.strip() == ground_truth.strip() else 0.0

    return rewards
''',

    "basic": '''import torch
from verl import DataProto

def compute_reward(data: DataProto, **kwargs) -> torch.Tensor:
    """Custom reward function."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        response = data.responses[i]
        ground_truth = data.get('ground_truths', [None])[i]

        if ground_truth is None:
            continue

        # YOUR SCORING LOGIC HERE
        rewards[i] = 1.0 if response == ground_truth else 0.0

    return rewards
''',
}


def main():
    parser = argparse.ArgumentParser(description="Create reward function template")
    parser.add_argument(
        "--task",
        choices=list(TEMPLATES.keys()),
        default="basic",
        help="Task type for reward function"
    )
    parser.add_argument(
        "--output",
        default="my_reward.py",
        help="Output file path"
    )

    args = parser.parse_args()

    template = TEMPLATES.get(args.task, TEMPLATES["basic"])

    with open(args.output, 'w') as f:
        f.write(template.strip() + '\n')

    print(f"✅ Created reward template: {args.output}")
    print(f"Task type: {args.task}")
    print()
    print("Next steps:")
    print(f"1. Edit {args.output} to implement your logic")
    print(f"2. Add to config:")
    print(f"   reward_model.custom_reward_function.path={args.output}")
    print(f"   reward_model.custom_reward_function.name=compute_reward")
    print(f"3. Run training with this reward function")


if __name__ == "__main__":
    main()
