"""Conditional metric wrapper."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from themis.core import entities as core_entities
from themis.interfaces import Metric


@dataclass
class ConditionalMetric:
    """Metric that only runs when condition is met.

    This wrapper allows you to conditionally apply metrics based on
    record characteristics (metadata, task type, etc.).

    Attributes:
        metric: Wrapped metric
        condition: Function that determines if metric should run
        default_score: Score to return when condition is False
        name: Optional override for metric name

    Example:
        >>> # Only run expensive metric on hard problems
        >>> hard_metric = ConditionalMetric(
        ...     metric=ExpensiveVerification(),
        ...     condition=lambda r: r.task.metadata.get("difficulty") == "hard",
        ...     default_score=0.0
        ... )
    """

    metric: Metric
    condition: Callable[[core_entities.GenerationRecord], bool]
    default_score: float = 0.0
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"conditional_{self.metric.name}"

    def should_evaluate(self, record: core_entities.GenerationRecord) -> bool:
        """Check if metric should be evaluated for this record.

        Args:
            record: Generation record

        Returns:
            True if condition is met
        """
        try:
            return self.condition(record)
        except Exception:
            # If condition check fails, don't run metric
            return False

    @property
    def requires_reference(self) -> bool:
        """Whether the wrapped metric requires references."""
        return getattr(self.metric, "requires_reference", True)

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        """Compute metric score.

        Note: This method doesn't check the condition - it's assumed
        the condition was already checked before calling compute.

        Args:
            prediction: Predicted value
            references: Reference values
            metadata: Optional metadata

        Returns:
            Metric score
        """
        return self.metric.compute(
            prediction=prediction,
            references=references,
            metadata=metadata,
        )

    def compute_or_default(
        self,
        record: core_entities.GenerationRecord,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        """Compute metric or return default score if condition not met.

        Args:
            record: Generation record (for condition check)
            prediction: Predicted value
            references: Reference values
            metadata: Optional metadata

        Returns:
            Metric score or default
        """
        if self.should_evaluate(record):
            return self.compute(
                prediction=prediction,
                references=references,
                metadata=metadata,
            )
        else:
            return core_entities.MetricScore(
                metric_name=self.name or self.metric.name,
                value=self.default_score,
                metadata={"skipped": True, "reason": "condition_not_met"},
            )
