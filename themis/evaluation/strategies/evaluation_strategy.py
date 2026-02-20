from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

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
        scores: list[core_entities.MetricScore],
    ) -> list[core_entities.MetricScore]:  # pragma: no cover - interface
        ...


__all__ = ["EvaluationStrategy"]
