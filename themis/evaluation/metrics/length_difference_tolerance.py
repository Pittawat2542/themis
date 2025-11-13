from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
from themis.interfaces import Metric as MetricInterface


@dataclass
class LengthDifferenceTolerance(MetricInterface):
    max_delta: int = 0

    def __post_init__(self) -> None:
        self.name = "LengthDifferenceTolerance"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        metadata = dict(metadata or {})
        reference = str(references[0]) if references else ""
        delta = abs(len(str(prediction)) - len(reference))
        value = 1.0 if delta <= self.max_delta else 0.0
        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details={"delta": delta},
            metadata=metadata,
        )
