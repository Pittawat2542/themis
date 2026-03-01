"""Base interfaces for progress reporting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProgressReporter(ABC):
    """Abstract base class for progress reporting."""

    @abstractmethod
    def __enter__(self) -> ProgressReporter:
        """Start the progress reporter."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the progress reporter."""
        pass

    @abstractmethod
    def add_task(self, description: str, total: int | None = None) -> int:
        """Add a new task to track.

        Returns:
            Task ID
        """
        pass

    @abstractmethod
    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        """Update task progress."""
        pass
