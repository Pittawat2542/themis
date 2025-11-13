from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from themis.core import entities as core_entities


@dataclass
class DefaultEvaluationStrategy:
    """Single-item evaluation for exact-match style metrics."""

    def prepare(
        self, record: core_entities.GenerationRecord
    ) -> Iterable[core_entities.EvaluationItem]:
        yield core_entities.EvaluationItem(
            record=record, reference=record.task.reference
        )

    def aggregate(
        self,
        record: core_entities.GenerationRecord,
        scores: List[core_entities.MetricScore],
    ) -> List[core_entities.MetricScore]:
        return scores
