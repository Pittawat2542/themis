"""Helper functions for common metric selection patterns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from themis.core import entities as core_entities
from themis.interfaces import Metric


def select_by_metadata_field(
    field: str, metric_map: dict[Any, list[Metric]], default: list[Metric] | None = None
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Create selector that chooses metrics based on metadata field value.

    Args:
        field: Metadata field to check
        metric_map: Mapping from field value to metrics
        default: Default metrics if field value not in map

    Returns:
        Metric selector function

    Example:
        >>> selector = select_by_metadata_field(
        ...     "type",
        ...     {
        ...         "math": [ExactMatch(), MathVerifyAccuracy()],
        ...         "code": [CodeExecutionMetric()],
        ...     },
        ...     default=[ExactMatch()]
        ... )
    """
    default_metrics = default or []

    def selector(record: core_entities.GenerationRecord) -> list[Metric]:
        value = record.task.metadata.get(field)
        return metric_map.get(value, default_metrics)

    return selector


def select_by_difficulty(
    easy_metrics: list[Metric],
    medium_metrics: list[Metric],
    hard_metrics: list[Metric],
    difficulty_field: str = "difficulty",
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Create selector that chooses metrics based on difficulty.

    Args:
        easy_metrics: Metrics for easy problems
        medium_metrics: Metrics for medium problems
        hard_metrics: Metrics for hard problems
        difficulty_field: Name of difficulty field in metadata

    Returns:
        Metric selector function

    Example:
        >>> selector = select_by_difficulty(
        ...     easy_metrics=[ExactMatch()],
        ...     medium_metrics=[ExactMatch(), PartialCredit()],
        ...     hard_metrics=[ExactMatch(), PartialCredit(), ManualReview()]
        ... )
    """
    return select_by_metadata_field(
        difficulty_field,
        {
            "easy": easy_metrics,
            "medium": medium_metrics,
            "hard": hard_metrics,
        },
        default=medium_metrics,
    )


def select_by_condition(
    condition: Callable[[core_entities.GenerationRecord], bool],
    metrics_if_true: list[Metric],
    metrics_if_false: list[Metric],
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Create selector based on arbitrary condition.

    Args:
        condition: Function to determine which metrics to use
        metrics_if_true: Metrics if condition is True
        metrics_if_false: Metrics if condition is False

    Returns:
        Metric selector function

    Example:
        >>> selector = select_by_condition(
        ...     lambda r: len(r.output.text) > 1000,
        ...     metrics_if_true=[SummaryMetrics()],
        ...     metrics_if_false=[ExactMatch()]
        ... )
    """

    def selector(record: core_entities.GenerationRecord) -> list[Metric]:
        try:
            if condition(record):
                return metrics_if_true
            else:
                return metrics_if_false
        except Exception:
            # If condition fails, use false branch
            return metrics_if_false

    return selector


def combine_selectors(
    *selectors: Callable[[core_entities.GenerationRecord], list[Metric]],
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Combine multiple selectors (union of their metrics).

    Args:
        *selectors: Metric selectors to combine

    Returns:
        Combined selector that returns union of all selected metrics

    Example:
        >>> selector = combine_selectors(
        ...     select_by_type,
        ...     select_by_difficulty,
        ... )
    """

    def combined(record: core_entities.GenerationRecord) -> list[Metric]:
        all_metrics = []
        seen_names = set()

        for selector in selectors:
            selected = selector(record)
            for metric in selected:
                if metric.name not in seen_names:
                    all_metrics.append(metric)
                    seen_names.add(metric.name)

        return all_metrics

    return combined
