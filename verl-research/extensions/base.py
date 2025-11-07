"""
Base classes for verl extensions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExtension(ABC):
    """
    Base class for all verl extensions.

    Extensions allow you to modify verl behavior without
    changing the original code.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize extension.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def apply(self, *args, **kwargs):
        """
        Apply the extension.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"
