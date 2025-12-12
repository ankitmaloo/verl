"""
sv2/data.py - Data loading utilities for multi-turn rollouts.

Provides functions to:
1. Build tokenizer/processor from model path
2. Create RLHFDataset from parquet files
3. Create DataLoader with proper collation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from verl.trainer.main_ppo import create_rl_dataset
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer


def build_tokenizer_processor(config: DictConfig) -> tuple[PreTrainedTokenizer, Any]:
    """
    Build tokenizer and processor from model config.

    Args:
        config: Full config with actor_rollout_ref.model.path

    Returns:
        Tuple of (tokenizer, processor)
    """
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    return tokenizer, processor


def create_dataset(
    data_paths: str | list[str],
    config: DictConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Any,
    is_train: bool = True,
    max_samples: int = -1,
) -> Dataset:
    """
    Create an RLHFDataset from data paths.

    Args:
        data_paths: Path(s) to parquet files
        config: Full config with data section
        tokenizer: HF tokenizer
        processor: HF processor (for multimodal)
        is_train: Whether this is training data
        max_samples: Max samples to load (-1 for all)

    Returns:
        Dataset instance
    """
    return create_rl_dataset(
        data_paths=data_paths,
        data_config=config.data,
        tokenizer=tokenizer,
        processor=processor,
        is_train=is_train,
        max_samples=max_samples,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with proper collation for RLHF data.

    Args:
        dataset: The dataset to load from
        batch_size: Batch size
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def select_data_paths(config: DictConfig, split: str) -> Any:
    """
    Select data paths based on split.

    Args:
        config: Full config with data section
        split: "train" or "val"

    Returns:
        Data file paths
    """
    if split == "train":
        return config.data.train_files
    if split == "val":
        return config.data.val_files
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")
