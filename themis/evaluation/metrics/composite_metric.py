from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
from themis.interfaces import Metric as MetricInterface


@dataclass
class CompositeMetric(MetricInterface):
    children: Sequence[MetricInterface]

    def __post_init__(self) -> None:
        self.name = "CompositeMetric"
        self.requires_reference = any(
            getattr(child, "requires_reference", True) for child in self.children
        )

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        child_results = [
            child.compute(
                prediction=prediction, references=references, metadata=metadata
            )
            for child in self.children
        ]
        if not child_results:
            return core_entities.MetricScore(
                metric_name=self.name,
                value=0.0,
                details={},
                metadata=dict(metadata or {}),
            )
        value = sum(result.value for result in child_results) / len(child_results)
        details = {result.metric_name: result.details for result in child_results}
        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details=details,
            metadata=dict(metadata or {}),
        )
