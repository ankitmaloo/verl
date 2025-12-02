"""Dataset classes for iterating through training data."""

import random
from base import GameDataset


class PasswordDataset(GameDataset):
    """Generates PasswordGame instances (each is unique due to random captcha/country)."""

    def get_batch(self, size: int) -> list:
        from pg import PasswordGame
        return [PasswordGame() for _ in range(size)]

    def __len__(self):
        return float('inf')


class GSM8KDataset(GameDataset):
    """
    Iterates through GSM8K dataset.

    Usage:
        dataset = GSM8KDataset("train", shuffle=True)
        for epoch in range(num_epochs):
            dataset.reset()  # reshuffle at epoch start
            for batch_idx in range(len(dataset) // batch_size):
                games = dataset.get_batch(batch_size)
                # train on batch...
    """

    def __init__(self, split="train", shuffle=True):
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        self.data = list(ds[split])
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.data)
        self.index = 0

    def __len__(self):
        return len(self.data)

    def get_batch(self, size: int) -> list:
        from gsm8k import GSM8KGame
        games = []
        for _ in range(size):
            problem = self.data[self.index % len(self.data)]
            games.append(GSM8KGame(problem))
            self.index += 1
        return games

    def reset(self):
        """Reset to beginning, optionally reshuffle."""
        self.index = 0
        if self.shuffle:
            random.shuffle(self.data)


DATASETS = {
    "password": PasswordDataset,
    "gsm8k": lambda: GSM8KDataset("train"),
    "gsm8k_test": lambda: GSM8KDataset("test", shuffle=False),
}


def get_dataset(name: str) -> GameDataset:
    """Get dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    factory = DATASETS[name]
    return factory() if callable(factory) else factory
