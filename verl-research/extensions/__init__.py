"""
Extension system for verl-research.

This module provides base classes for extending verl functionality
without modifying the original codebase.
"""

from .base import BaseExtension
from .custom_advantages import BaseAdvantageCompute
from .custom_losses import BaseLossCompute
from .custom_rewards import BaseRewardShaper
from .custom_samplers import BaseSampler

__all__ = [
    'BaseExtension',
    'BaseAdvantageCompute',
    'BaseLossCompute',
    'BaseRewardShaper',
    'BaseSampler',
]
