"""Batch optimization for generation.

This module provides utilities for batching generation tasks to improve efficiency:
- BatchConfig: Configuration for batching behavior
- TaskBatcher: Groups tasks for efficient batch processing
- Batch-aware runner patterns

Batching can reduce:
- API call overhead
- Network latency
- Total generation time
- Cost (for providers with batch APIs)

Example:
    >>> config = BatchConfig(max_batch_size=10, group_by=lambda t: t.model.identifier)
    >>> batcher = TaskBatcher(config)
    >>>
    >>> # Group tasks
    >>> for batch in batcher.create_batches(tasks):
    ...     results = provider.generate_batch(batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from themis.core import entities as core_entities


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        max_batch_size: Maximum number of tasks per batch
        group_by: Function to group compatible tasks (same return value = same batch)
        timeout_ms: Maximum time to wait for batch to fill (future use)
    """

    max_batch_size: int = 10
    group_by: Callable[[core_entities.GenerationTask], str] | None = None
    timeout_ms: float = 100

    def __post_init__(self):
        """Validate configuration."""
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.timeout_ms < 0:
            raise ValueError("timeout_ms must be >= 0")


class TaskBatcher:
    """Groups generation tasks into batches for efficient processing.

    The batcher can group tasks by various criteria (model, prompt length, etc.)
    and create batches within size limits.

    Example:
        >>> batcher = TaskBatcher(BatchConfig(max_batch_size=5))
        >>> tasks = [...]  # 20 tasks
        >>> batches = list(batcher.create_batches(tasks))
        >>> len(batches)  # 4 batches of 5 each
        4
    """

    def __init__(self, config: BatchConfig):
        """Initialize batcher.

        Args:
            config: Batch configuration
        """
        self._config = config

    def create_batches(
        self, tasks: Sequence[core_entities.GenerationTask]
    ) -> Iterator[list[core_entities.GenerationTask]]:
        """Create batches from tasks.

        If group_by is specified, groups tasks first, then creates batches within groups.
        Otherwise, creates batches in order up to max_batch_size.

        Args:
            tasks: Tasks to batch

        Yields:
            Batches of tasks
        """
        if self._config.group_by is None:
            # Simple batching without grouping
            yield from self._batch_sequential(tasks)
        else:
            # Group tasks first, then batch each group
            groups = self._group_tasks(tasks)
            for group in groups.values():
                yield from self._batch_sequential(group)

    def _group_tasks(
        self, tasks: Sequence[core_entities.GenerationTask]
    ) -> dict[str, list[core_entities.GenerationTask]]:
        """Group tasks by grouping function.

        Args:
            tasks: Tasks to group

        Returns:
            Dictionary mapping group keys to task lists
        """
        if self._config.group_by is None:
            return {"default": list(tasks)}

        groups: dict[str, list[core_entities.GenerationTask]] = {}
        for task in tasks:
            key = self._config.group_by(task)
            groups.setdefault(key, []).append(task)

        return groups

    def _batch_sequential(
        self, tasks: Sequence[core_entities.GenerationTask]
    ) -> Iterator[list[core_entities.GenerationTask]]:
        """Create batches sequentially up to max_batch_size.

        Args:
            tasks: Tasks to batch

        Yields:
            Batches of tasks
        """
        batch = []
        for task in tasks:
            batch.append(task)
            if len(batch) >= self._config.max_batch_size:
                yield batch
                batch = []

        # Yield remaining tasks
        if batch:
            yield batch

    def get_batch_count(self, tasks: Sequence[core_entities.GenerationTask]) -> int:
        """Get number of batches that would be created.

        Args:
            tasks: Tasks to batch

        Returns:
            Number of batches
        """
        return len(list(self.create_batches(tasks)))

    def get_batch_stats(
        self, tasks: Sequence[core_entities.GenerationTask]
    ) -> dict[str, Any]:
        """Get statistics about batching.

        Args:
            tasks: Tasks to batch

        Returns:
            Dictionary with batching statistics
        """
        batches = list(self.create_batches(tasks))
        batch_sizes = [len(b) for b in batches]

        stats = {
            "total_tasks": len(tasks),
            "num_batches": len(batches),
            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            "min_batch_size": min(batch_sizes) if batch_sizes else 0,
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
        }

        # Add group stats if grouping is enabled
        if self._config.group_by is not None:
            groups = self._group_tasks(tasks)
            stats["num_groups"] = len(groups)
            stats["group_sizes"] = {key: len(tasks) for key, tasks in groups.items()}

        return stats


# ============================================================================
# Batch-Aware Helpers
# ============================================================================


def group_by_model(task: core_entities.GenerationTask) -> str:
    """Group tasks by model identifier.

    Args:
        task: Task to group

    Returns:
        Model identifier as grouping key
    """
    return task.model.identifier


def group_by_prompt_length(
    task: core_entities.GenerationTask, bucket_size: int = 100
) -> str:
    """Group tasks by prompt length (bucketed).

    Groups tasks into buckets based on prompt length. This can help
    optimize batch processing when prompt length affects performance.

    Args:
        task: Task to group
        bucket_size: Size of length buckets

    Returns:
        Bucket identifier as grouping key
    """
    length = len(task.prompt.text)
    bucket = (length // bucket_size) * bucket_size
    return f"length_{bucket}-{bucket + bucket_size}"


def group_by_model_and_sampling(task: core_entities.GenerationTask) -> str:
    """Group tasks by model and sampling configuration.

    Args:
        task: Task to group

    Returns:
        Combined model and sampling key
    """
    sampling_key = f"t{task.sampling.temperature}_p{task.sampling.top_p}"
    return f"{task.model.identifier}_{sampling_key}"


def create_grouping_function(
    *groupers: Callable[[core_entities.GenerationTask], str],
) -> Callable[[core_entities.GenerationTask], str]:
    """Create a composite grouping function from multiple groupers.

    Args:
        *groupers: Grouping functions to combine

    Returns:
        Combined grouping function

    Example:
        >>> # Group by both model and prompt length
        >>> grouper = create_grouping_function(group_by_model, group_by_prompt_length)
        >>> config = BatchConfig(group_by=grouper)
    """

    def combined_grouper(task: core_entities.GenerationTask) -> str:
        keys = [grouper(task) for grouper in groupers]
        return "_".join(keys)

    return combined_grouper
