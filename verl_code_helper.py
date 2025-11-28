"""
VERL Code Writing Helper Module

Provides utilities and templates for common VERL modifications:
- Custom reward functions
- Custom loss functions
- Custom datasets
- Custom samplers
- Configuration management

Usage:
    from verl_code_helper import RewardFunctionTemplate, LossFunctionTemplate

    # Create a template
    template = RewardFunctionTemplate.for_math_problems()
    template.save("my_reward.py")

    # Or use directly
    from verl_code_helper import create_simple_reward_fn
    reward_fn = create_simple_reward_fn(task="math")
"""

import os
import inspect
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from enum import Enum


class TaskType(Enum):
    """Common VERL task types."""
    MATH = "math"
    CODE = "code"
    QA = "qa"
    CONVERSATION = "conversation"
    RANKING = "ranking"


class ComponentType(Enum):
    """VERL pipeline components."""
    INFERENCE = "inference"
    TRAINING = "training"
    ENVIRONMENT = "environment"
    ALGORITHM = "algorithm"


class RewardFunctionTemplate:
    """Template for creating custom reward functions."""

    BASIC_TEMPLATE = '''
import torch
from verl import DataProto
from typing import Optional


def compute_reward(
    data: DataProto,
    tokenizer=None,
    **kwargs
) -> torch.Tensor:
    """
    Compute rewards for generated responses.

    Args:
        data: DataProto with batch information
        tokenizer: Optional tokenizer for post-processing
        **kwargs: Additional reward configuration

    Returns:
        torch.Tensor: Reward tensor of shape [batch_size]
    """
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    # Get responses and ground truths
    responses = data.responses  # List of strings
    ground_truths = data.get('ground_truths', [None] * batch_size)

    for i in range(batch_size):
        response = responses[i]
        ground_truth = ground_truths[i]

        # Implement your scoring logic
        score = compute_single_score(response, ground_truth, **kwargs)
        rewards[i] = score

    return rewards


def compute_single_score(response: str, ground_truth: Optional[str], **kwargs) -> float:
    """Score a single response."""
    if ground_truth is None:
        return 0.0

    # YOUR IMPLEMENTATION HERE
    # Example: exact match
    return 1.0 if response.strip() == ground_truth.strip() else 0.0
'''

    MATH_TEMPLATE = '''
import torch
import re
from verl import DataProto
from typing import Optional


def compute_reward(
    data: DataProto,
    tokenizer=None,
    **kwargs
) -> torch.Tensor:
    """Compute rewards for math problems."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    responses = data.responses
    ground_truths = data.get('ground_truths', [None] * batch_size)

    for i in range(batch_size):
        response = responses[i]
        ground_truth = ground_truths[i]

        if ground_truth is None:
            continue

        # Extract numeric answers
        response_ans = extract_answer(response)
        truth_ans = extract_answer(ground_truth)

        # Check correctness (with tolerance for floating point)
        if response_ans is not None and truth_ans is not None:
            if abs(response_ans - truth_ans) < 1e-6:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        else:
            rewards[i] = 0.0

    return rewards


def extract_answer(text: str) -> Optional[float]:
    """Extract numeric answer from text."""
    try:
        # Look for patterns like "The answer is 42" or "= 42"
        patterns = [
            r'answer\s*(?:is|:)\s*([-+]?\\d+\\.?\\d*)',
            r'(?:=|≈)\s*([-+]?\\d+\\.?\\d*)',
            r'([-+]?\\d+\\.?\\d*)\\s*(?:is\\s+)?(?:the\\s+)?answer',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Last resort: find any number
        numbers = re.findall(r'[-+]?\\d+\\.?\\d*', text)
        if numbers:
            return float(numbers[-1])
    except:
        pass

    return None
'''

    CODE_TEMPLATE = '''
import torch
from verl import DataProto
from typing import Optional
import subprocess
import tempfile


def compute_reward(
    data: DataProto,
    tokenizer=None,
    **kwargs
) -> torch.Tensor:
    """Compute rewards for code generation tasks."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    responses = data.responses
    test_cases = data.get('test_cases', [None] * batch_size)

    for i in range(batch_size):
        code = responses[i]
        tests = test_cases[i]

        if tests is None:
            continue

        # Execute code and test
        passed = execute_code_with_tests(code, tests)
        rewards[i] = 1.0 if passed else 0.0

    return rewards


def execute_code_with_tests(code: str, test_cases: List[Dict]) -> bool:
    """Execute code and check if it passes test cases."""
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name

        # Run tests
        for test in test_cases:
            input_data = test['input']
            expected = test['output']

            # Execute and compare (simplified)
            # In practice, you'd need proper sandboxing
            result = subprocess.run(
                ['python', code_file],
                input=input_data,
                capture_output=True,
                timeout=5,
                text=True
            )

            if result.stdout.strip() != expected.strip():
                return False

        return True
    except Exception as e:
        print(f"Error executing code: {e}")
        return False
'''

    QA_TEMPLATE = '''
import torch
from verl import DataProto
from typing import Optional


def compute_reward(
    data: DataProto,
    tokenizer=None,
    use_partial_credit=True,
    **kwargs
) -> torch.Tensor:
    """Compute rewards for QA tasks."""
    batch_size = len(data.batch)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    responses = data.responses
    ground_truths = data.get('ground_truths', [None] * batch_size)

    for i in range(batch_size):
        response = responses[i]
        ground_truth = ground_truths[i]

        if ground_truth is None:
            continue

        if use_partial_credit:
            # Use BLEU/ROUGE-like score
            score = compute_similarity(response, ground_truth)
            rewards[i] = score
        else:
            # Exact match only
            rewards[i] = 1.0 if normalize(response) == normalize(ground_truth) else 0.0

    return rewards


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return ' '.join(text.lower().split())


def compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, normalize(text1), normalize(text2)).ratio()
'''

    @classmethod
    def for_task(cls, task_type: TaskType) -> "RewardFunctionTemplate":
        """Get template for specific task type."""
        templates = {
            TaskType.MATH: cls.MATH_TEMPLATE,
            TaskType.CODE: cls.CODE_TEMPLATE,
            TaskType.QA: cls.QA_TEMPLATE,
            TaskType.CONVERSATION: cls.BASIC_TEMPLATE,
            TaskType.RANKING: cls.BASIC_TEMPLATE,
        }

        template = RewardFunctionTemplate()
        template.code = templates.get(task_type, cls.BASIC_TEMPLATE)
        template.task_type = task_type
        return template

    @classmethod
    def for_math_problems(cls) -> "RewardFunctionTemplate":
        """Template for math problem scoring."""
        return cls.for_task(TaskType.MATH)

    @classmethod
    def for_code_generation(cls) -> "RewardFunctionTemplate":
        """Template for code generation scoring."""
        return cls.for_task(TaskType.CODE)

    @classmethod
    def for_qa(cls) -> "RewardFunctionTemplate":
        """Template for QA scoring."""
        return cls.for_task(TaskType.QA)

    def __init__(self):
        self.code = self.BASIC_TEMPLATE
        self.task_type = None

    def save(self, file_path: str) -> None:
        """Save template to file."""
        with open(file_path, 'w') as f:
            f.write(textwrap.dedent(self.code).strip())
        print(f"Saved reward function template to {file_path}")

    def get_config(self) -> Dict[str, str]:
        """Get required config changes."""
        return {
            "reward_model.custom_reward_function.path": Path.cwd().joinpath("my_reward.py").as_posix(),
            "reward_model.custom_reward_function.name": "compute_reward",
        }

    def __str__(self) -> str:
        return textwrap.dedent(self.code).strip()


class LossFunctionTemplate:
    """Template for creating custom loss functions."""

    BASIC_PPO = '''
import torch
from typing import Optional, Dict, Any


def my_custom_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token_level",
    config: Optional[Any] = None,
    rollout_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Custom PPO policy loss with modifications.

    Args:
        old_log_probs: Log probs from reference [batch, seq_len]
        log_probs: Log probs from actor [batch, seq_len]
        advantages: Advantage estimates [batch, seq_len]
        response_mask: Valid token mask [batch, seq_len]
        loss_agg_mode: Aggregation mode
        config: Actor configuration
        rollout_log_probs: Optional rollout log probs

    Returns:
        loss: Scalar loss tensor
        loss_info: Dict with loss components
    """
    # Compute probability ratio
    ratio = (log_probs - old_log_probs).exp()

    # PPO clipped objective
    epsilon = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)

    # Apply mask
    policy_loss = policy_loss * response_mask

    # Aggregate
    if loss_agg_mode == "token_level":
        loss = policy_loss.sum() / response_mask.sum().clamp(min=1)
    else:
        loss = policy_loss.mean()

    return loss, {
        "policy_loss": policy_loss.mean().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
    }
'''

    @classmethod
    def basic_ppo(cls) -> "LossFunctionTemplate":
        """Get basic PPO loss template."""
        template = LossFunctionTemplate()
        template.code = cls.BASIC_PPO
        return template

    def __init__(self):
        self.code = self.BASIC_PPO

    def save(self, file_path: str) -> None:
        """Save to file. Note: this should be added to core_algos.py with @register_policy_loss decorator."""
        with open(file_path, 'w') as f:
            f.write(textwrap.dedent(self.code).strip())
        print(f"Saved loss function template to {file_path}")
        print("Note: Add @register_policy_loss('my_loss') decorator to core_algos.py")

    def get_config(self) -> Dict[str, str]:
        """Get required config changes."""
        return {
            "algorithm.policy_loss_fn": "my_custom_policy_loss",
        }

    def __str__(self) -> str:
        return textwrap.dedent(self.code).strip()


class DatasetTemplate:
    """Template for creating custom datasets."""

    BASIC = '''
import json
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict


class MyDataset(Dataset):
    """Custom dataset for VERL training."""

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        processor=None,
        config=None,
        **kwargs
    ):
        """
        Args:
            data_files: List of file paths
            tokenizer: HuggingFace tokenizer
            processor: Optional image processor for multimodal
            config: Data configuration
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or {}
        self.data = []

        # Load data from files
        for file_path in data_files:
            self._load_file(file_path)

    def _load_file(self, file_path: str):
        """Load data from a file."""
        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                if file_path.endswith('.jsonl'):
                    items = [json.loads(line) for line in f]
                else:
                    items = json.load(f)
                self.data.extend(items if isinstance(items, list) else [items])
        elif file_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(file_path)
            self.data.extend(df.to_dict('records'))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Extract fields
        prompt = item.get('prompt', item.get('instruction', ''))
        response = item.get('response', item.get('output', ''))

        # Tokenize
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.get('max_prompt_length', 512),
        )

        response_tokens = self.tokenizer.encode(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.get('max_response_length', 1024),
        )

        return {
            'input_ids': torch.tensor(prompt_tokens, dtype=torch.long),
            'response_ids': torch.tensor(response_tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(prompt_tokens), dtype=torch.long),
        }
'''

    @classmethod
    def basic(cls) -> "DatasetTemplate":
        """Get basic dataset template."""
        template = DatasetTemplate()
        template.code = cls.BASIC
        return template

    def __init__(self):
        self.code = self.BASIC

    def save(self, file_path: str) -> None:
        """Save to file."""
        with open(file_path, 'w') as f:
            f.write(textwrap.dedent(self.code).strip())
        print(f"Saved dataset template to {file_path}")

    def get_config(self) -> Dict[str, str]:
        """Get required config changes."""
        return {
            "data.custom_cls.path": Path.cwd().joinpath("my_dataset.py").as_posix(),
            "data.custom_cls.name": "MyDataset",
        }

    def __str__(self) -> str:
        return textwrap.dedent(self.code).strip()


class ConfigBuilder:
    """Builder for creating VERL configuration."""

    def __init__(self):
        self.config_dict = {}

    def set_inference_engine(self, engine: str) -> "ConfigBuilder":
        """Set inference engine (sglang, vllm, hf)."""
        self.config_dict['actor_rollout_ref.rollout.name'] = engine
        return self

    def set_advantage_estimator(self, estimator: str) -> "ConfigBuilder":
        """Set advantage estimator (gae, grpo, reinforce_plus_plus, etc)."""
        self.config_dict['algorithm.adv_estimator'] = estimator
        return self

    def set_learning_rate(self, lr: float) -> "ConfigBuilder":
        """Set actor learning rate."""
        self.config_dict['actor_rollout_ref.actor.optim.lr'] = lr
        return self

    def set_data_files(self, train_files: str, val_files: str) -> "ConfigBuilder":
        """Set data file paths."""
        self.config_dict['data.train_files'] = train_files
        self.config_dict['data.val_files'] = val_files
        return self

    def set_reward_function(self, path: str, name: str) -> "ConfigBuilder":
        """Set custom reward function."""
        self.config_dict['reward_model.custom_reward_function.path'] = path
        self.config_dict['reward_model.custom_reward_function.name'] = name
        return self

    def set_dataset_class(self, path: str, name: str) -> "ConfigBuilder":
        """Set custom dataset class."""
        self.config_dict['data.custom_cls.path'] = path
        self.config_dict['data.custom_cls.name'] = name
        return self

    def set_batch_size(self, batch_size: int) -> "ConfigBuilder":
        """Set training batch size."""
        self.config_dict['data.train_batch_size'] = batch_size
        return self

    def add_config(self, key: str, value: Any) -> "ConfigBuilder":
        """Add arbitrary config option."""
        self.config_dict[key] = value
        return self

    def build_cli_args(self) -> str:
        """Build command-line arguments string."""
        args = []
        for key, value in self.config_dict.items():
            if isinstance(value, str):
                args.append(f"{key}={value}")
            elif isinstance(value, bool):
                args.append(f"{key}={str(value).lower()}")
            else:
                args.append(f"{key}={value}")
        return " \\\n    ".join(args)

    def build_shell_script(self, script_path: str = "run_training.sh") -> str:
        """Generate a shell script with these config options."""
        cli_args = self.build_cli_args()
        script = f'''#!/bin/bash

# Generated VERL training script
# {len(self.config_dict)} configuration options

python3 -m verl.trainer.main_ppo \\
    {cli_args}
'''
        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        print(f"Saved training script to {script_path}")
        return script

    def __str__(self) -> str:
        """Return as CLI arguments."""
        return self.build_cli_args()


def create_reward_function_for_task(task: TaskType) -> str:
    """Create and return reward function code for a task."""
    template = RewardFunctionTemplate.for_task(task)
    return str(template)


def create_example_training_setup() -> None:
    """Create example training setup with custom components."""
    print("Creating example training setup...")

    # Create reward function
    reward_template = RewardFunctionTemplate.for_math_problems()
    reward_template.save("my_reward.py")

    # Create dataset
    dataset_template = DatasetTemplate.basic()
    dataset_template.save("my_dataset.py")

    # Create config
    config = ConfigBuilder()
    config.set_inference_engine("sglang") \
        .set_advantage_estimator("grpo") \
        .set_learning_rate(1e-6) \
        .set_data_files("~/data/train.parquet", "~/data/test.parquet") \
        .set_reward_function("my_reward.py", "compute_reward") \
        .set_dataset_class("my_dataset.py", "MyDataset") \
        .set_batch_size(1024)

    config.build_shell_script("run_training.sh")

    print(f"""
Example training setup created!

Files created:
  - my_reward.py          (custom reward function)
  - my_dataset.py         (custom dataset class)
  - run_training.sh       (training script)

Next steps:
  1. Edit my_reward.py to implement your scoring logic
  2. Edit my_dataset.py to load your data format
  3. Update data file paths in run_training.sh
  4. Run: bash run_training.sh
    """)


# Convenience functions
def get_reward_template(task: str) -> RewardFunctionTemplate:
    """Get reward template by task name string."""
    task_map = {
        'math': TaskType.MATH,
        'code': TaskType.CODE,
        'qa': TaskType.QA,
        'conversation': TaskType.CONVERSATION,
        'ranking': TaskType.RANKING,
    }
    task_type = task_map.get(task.lower(), TaskType.QA)
    return RewardFunctionTemplate.for_task(task_type)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("VERL Code Writing Helper - Examples")
    print("=" * 60)

    # Example 1: Create reward function
    print("\n1. Creating math problem reward function...")
    reward_template = RewardFunctionTemplate.for_math_problems()
    print(reward_template)

    # Example 2: Create config
    print("\n2. Creating training configuration...")
    config = ConfigBuilder()
    config.set_inference_engine("sglang") \
        .set_advantage_estimator("grpo") \
        .set_learning_rate(1e-6)
    print(config)

    # Example 3: Create full setup
    print("\n3. Creating full training setup...")
    create_example_training_setup()
