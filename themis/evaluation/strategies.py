"""Evaluation task strategies and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class AttemptAwareEvaluationStrategy:
    """Evaluates each generation attempt independently."""

    average_attempts: bool = True

    def prepare(
        self, record: core_entities.GenerationRecord
    ) -> Iterable[core_entities.EvaluationItem]:
        attempts = record.attempts or [record]
        for attempt in attempts:
            yield core_entities.EvaluationItem(
                record=attempt, reference=attempt.task.reference
            )

    def aggregate(
        self,
        record: core_entities.GenerationRecord,
        scores: List[core_entities.MetricScore],
    ) -> List[core_entities.MetricScore]:
        if not self.average_attempts or not scores:
            return scores
        aggregated = []
        grouped: dict[str, list[core_entities.MetricScore]] = {}
        for score in scores:
            grouped.setdefault(score.metric_name, []).append(score)
        for metric_name, group in grouped.items():
            value = sum(item.value for item in group) / len(group)
            aggregated.append(
                core_entities.MetricScore(
                    metric_name=metric_name,
                    value=value,
                    metadata={
                        "attempts": len(group),
                        "sample_id": group[0].metadata.get("sample_id"),
                    },
                    details={},
                )
            )
        return aggregated


__all__ = [
    "EvaluationStrategy",
    "DefaultEvaluationStrategy",
    "AttemptAwareEvaluationStrategy",
]
