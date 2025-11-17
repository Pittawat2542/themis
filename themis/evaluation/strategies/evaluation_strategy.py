from __future__ import annotations

from typing import Iterable, List, Protocol

from themis.core import entities as core_entities


class EvaluationStrategy(Protocol):
    """Strategy controlling how evaluation items are constructed and aggregated."""

    def prepare(
        self, record: core_entities.GenerationRecord
    ) -> Iterable[core_entities.EvaluationItem]:  # pragma: no cover - interface
        ...

    def aggregate(
        self,
        record: core_entities.GenerationRecord,
        scores: List[core_entities.MetricScore],
    ) -> List[core_entities.MetricScore]:  # pragma: no cover - interface
        ...


__all__ = ["EvaluationStrategy"]
