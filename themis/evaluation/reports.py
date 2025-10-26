"""Evaluation report data structures."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List

from themis.core import entities as core_entities


@dataclass
class EvaluationFailure:
    sample_id: str | None
    message: str


@dataclass
class MetricAggregate:
    name: str
    count: int
    mean: float
    per_sample: List[core_entities.MetricScore]

    @classmethod
    def from_scores(
        cls, name: str, scores: List[core_entities.MetricScore]
    ) -> "MetricAggregate":
        if not scores:
            return cls(name=name, count=0, mean=0.0, per_sample=[])
        return cls(
            name=name,
            count=len(scores),
            mean=mean(score.value for score in scores),
            per_sample=scores,
        )


@dataclass
class EvaluationReport:
    metrics: dict[str, MetricAggregate]
    failures: List[EvaluationFailure]
    records: List[core_entities.EvaluationRecord]


__all__ = [
    "EvaluationFailure",
    "MetricAggregate",
    "EvaluationReport",
]
