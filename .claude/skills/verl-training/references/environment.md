# Environment: Data Pipeline & Dataset Configuration

How to load data, create custom datasets, and implement curriculum learning strategies.

## Table of Contents

1. Dataset Loading
2. Creating Custom Datasets
3. Data Tokenization & Formatting
4. Curriculum Learning
5. Batch Size & Sampling
6. Dynamic Data Generation

## Dataset Loading

VERL loads data via `create_rl_dataset()` in `main_ppo.py:369`.

### Supported File Formats

**Parquet Files (Recommended)**
```bash
data.train_files=/path/to/train.parquet
data.val_files=/path/to/val.parquet
```
Efficient columnar format, good for large datasets.

**JSON Files**
```bash
data.train_files=/path/to/train.json
# Or JSONL (one object per line)
data.train_files=/path/to/train.jsonl
```

**Multiple Files**
```bash
data.train_files=[/path/file1.parquet, /path/file2.parquet]
```

### Dataset Selection Logic

The framework automatically selects dataset class:

```python
# In main_ppo.py:369-416
if config.data.custom_cls and config.data.custom_cls.get("path"):
    # Use custom dataset
    dataset_cls = load_extern_type(config.data.custom_cls.path, config.data.custom_cls.name)
elif config.data.datagen and config.data.datagen.get("path") and is_train:
    # Use dynamic data generation
    dataset_cls = DynamicGenDataset
else:
    # Use default RLHFDataset
    dataset_cls = RLHFDataset
```

## Creating Custom Datasets

Implement custom `torch.utils.data.Dataset`:

### Basic Template

```python
# File: my_dataset.py
import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Optional

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
            data_files: List of file paths (parquet, json, jsonl)
            tokenizer: HuggingFace tokenizer
            processor: Optional processor for multimodal (images)
            config: Data configuration from Hydra
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or {}
        self.data = []

        # Load all data files
        for file_path in data_files:
            self._load_file(file_path)

    def _load_file(self, file_path: str):
        """Load data from single file."""
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                items = json.load(f)
                if isinstance(items, dict):
                    items = [items]
                self.data.extend(items)

        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            self.data.extend(df.to_dict('records'))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Extract fields (adapt to your data format)
        prompt = item.get('prompt', item.get('instruction', ''))
        response = item.get('response', item.get('output', ''))

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.get('max_prompt_length', 512),
        )

        # Tokenize response
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
```

### Register Custom Dataset

```bash
data.custom_cls.path=my_dataset.py
data.custom_cls.name=MyDataset
```

### Advanced: Multi-Modal Dataset

```python
class MultiModalDataset(Dataset):
    """Dataset with images and text."""

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text
        prompt_tokens = self.tokenizer.encode(item['text'])

        # Process image if using multimodal model
        image = None
        if 'image_path' in item and self.processor:
            from PIL import Image
            image = Image.open(item['image_path'])
            image_inputs = self.processor(images=image, return_tensors="pt")

        return {
            'input_ids': torch.tensor(prompt_tokens),
            'pixel_values': image_inputs.get('pixel_values'),  # For vision models
            'image_seq_len': image_inputs.get('image_seq_len'),
        }
```

## Data Tokenization & Formatting

### Standard Fields

The RLHFDataset expects these columns (customize as needed):

```python
{
    'prompt': str,          # Input to model
    'response': str,        # Expected output
    # Optional fields:
    'instruction': str,     # Alternative to 'prompt'
    'output': str,          # Alternative to 'response'
    'ground_truth': any,    # For reward computation
}
```

### Custom Tokenization

```python
def __getitem__(self, idx):
    item = self.data[idx]

    # Pad prompt to consistent length
    prompt_tokens = self.tokenizer.encode(item['prompt'])
    prompt_tokens = self._pad_or_truncate(prompt_tokens, self.config['max_prompt_length'])

    # Add special tokens to response
    response_tokens = self.tokenizer.encode(
        item['response'],
        add_special_tokens=False,  # Avoid double-adding
    )

    return {
        'input_ids': torch.tensor(prompt_tokens),
        'response_ids': torch.tensor(response_tokens),
        'attention_mask': torch.ones_like(torch.tensor(prompt_tokens)),
    }

def _pad_or_truncate(self, tokens: List[int], max_len: int) -> List[int]:
    """Pad to max_len or truncate if too long."""
    if len(tokens) < max_len:
        tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens
```

### Data with Ground Truth Answers

```python
{
    'prompt': 'What is 2+2?',
    'response': 'The answer is 4.',
    'ground_truth': 4,  # For reward computation
}
```

Accessed in reward function:
```python
def compute_reward(data: DataProto, **kwargs):
    for i in range(len(data.batch)):
        response = data.responses[i]
        ground_truth = data.ground_truths[i]  # From dataset
        # Compare and score
```

## Curriculum Learning

Implement difficulty-based sampling order:

### Basic Curriculum Sampler

```python
# File: my_sampler.py
import torch
import numpy as np
from verl.experimental.dataset.sampler import AbstractSampler

class DifficultySampler(AbstractSampler):
    """Sample data by difficulty level."""

    def __init__(self, data_source, data_config, **kwargs):
        self.data_source = data_source
        self.config = data_config
        self.num_samples = len(data_source)

        # Initialize with uniform difficulty
        self.difficulty_scores = np.ones(self.num_samples)
        self.training_step = 0
        self.update_frequency = 10  # Update difficulty every N batches

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        """Sample indices based on curriculum difficulty."""
        # Normalize scores to probability distribution
        probs = self.difficulty_scores / self.difficulty_scores.sum()

        # Sample with replacement biased by difficulty
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            p=probs,
            replace=True,
        )

        return iter(indices.tolist())

    def update(self, indices: List[int], metrics: Dict):
        """Update difficulty scores based on training metrics."""
        for idx in indices:
            loss = metrics.get(idx, {}).get('loss', float('inf'))

            # Increase difficulty for easy samples (low loss)
            if loss < 0.5:
                self.difficulty_scores[idx] *= 1.1
            # Decrease difficulty for hard samples (high loss)
            elif loss > 2.0:
                self.difficulty_scores[idx] *= 0.9

        self.training_step += 1
```

### Register Curriculum Sampler

```bash
data.sampler.class_path=my_sampler.py
data.sampler.class_name=DifficultySampler

# IMPORTANT: Curriculum requires num_workers=0
data.dataloader_num_workers=0
```

### Advanced: Learning Rate Curriculum

```python
class LRCurriculumSampler(AbstractSampler):
    """Adjust learning rate based on data difficulty."""

    def update(self, indices, metrics):
        avg_loss = np.mean([m.get('loss', 1.0) for m in metrics.values()])

        # Increase LR if learning is stable (loss decreasing)
        if avg_loss < self.prev_loss:
            self.learning_rate *= 1.05
        # Decrease LR if training is unstable
        else:
            self.learning_rate *= 0.95

        self.prev_loss = avg_loss
```

## Batch Size & Sampling

### Configuration

```yaml
data:
  train_batch_size: 1024  # Total batch size (across all GPUs)
  dataloader_num_workers: 8  # Parallel data loading
  shuffle: true
  seed: 42
```

### Memory Optimization

**Reduce batch size:**
```bash
data.train_batch_size=512
```

**Per-GPU batch sizing:**
```bash
data.train_batch_size=256  # Total
# Per GPU = 256 / (n_gpus_per_node * nnodes)
```

### Sampling Modes

**Shuffle (Random):**
```yaml
data:
  shuffle: true
  seed: 42  # For reproducibility
```

**Sequential:**
```yaml
data:
  shuffle: false
```

**Curriculum (via sampler):**
```yaml
data:
  sampler:
    class_path: my_sampler.py
    class_name: DifficultySampler
  dataloader_num_workers: 0  # Required for curriculum
```

## Dynamic Data Generation

For online data generation during training:

```python
# File: my_datagen.py
from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

class OnlineCurriculumDataset(DynamicGenDataset):
    """Generate data based on training progress."""

    def generate_sample(self, global_step: int):
        """Generate a new sample dynamically."""
        # Difficulty increases with training steps
        difficulty = min(global_step / 1000, 1.0)

        # Generate problem with appropriate difficulty
        problem = generate_problem(difficulty)
        response = solve_problem(problem)

        return {
            'prompt': problem,
            'response': response,
            'difficulty': difficulty,
        }
```

### Config

```yaml
data:
  datagen:
    path: my_datagen.py
    name: OnlineCurriculumDataset
```

## Entry Point in Training

**File:** `verl/trainer/main_ppo.py:329-344`

```python
# Load training data
train_dataset = create_rl_dataset(
    config.data.train_files,
    config.data,
    tokenizer,
    processor,
    is_train=True,
)

# Create sampler
train_sampler = create_rl_sampler(config.data, train_dataset)

# DataLoader created in RayPPOTrainer
# Batches flow to training loop
```

**In Training Loop:** `ray_trainer.py:1024`

```python
for batch_dict in self.train_dataloader:
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    # Process batch...
```

Each `batch_dict` contains fields from dataset's `__getitem__()`.
