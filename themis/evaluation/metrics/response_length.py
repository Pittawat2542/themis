from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
from themis.interfaces import Metric as MetricInterface


@dataclass
class ResponseLength(MetricInterface):
    """Reports the length of the prediction response."""

    def __post_init__(self) -> None:
        self.name = "ResponseLength"
        self.requires_reference = False

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        metadata = dict(metadata or {})
        text = str(prediction)
        length = len(text)
        return core_entities.MetricScore(
            metric_name=self.name,
            value=float(length),
            details={"length": length},
            metadata=metadata,
        )
