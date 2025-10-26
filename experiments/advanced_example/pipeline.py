"""Custom evaluation pipeline extensions."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Sequence

from themis.core import entities as core_entities
from themis.evaluation import pipeline as base_pipeline


class SubjectAwareEvaluationPipeline(base_pipeline.EvaluationPipeline):
    """Extends the base evaluation pipeline by tracking per-subject aggregates."""

    def __init__(self, *args, subject_field: str = "subject", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._subject_field = subject_field
        self.subject_breakdown: dict[str, float] = {}

    def evaluate(self, records: Sequence[core_entities.GenerationRecord]):  # type: ignore[override]
        report = super().evaluate(records)
        if not records:
            self.subject_breakdown = {}
            return report

        subject_scores: defaultdict[str, list[float]] = defaultdict(list)
        score_lookup: dict[str, float] = {}
        exact_metric = report.metrics.get("ExactMatch")
        if exact_metric:
            for score in exact_metric.per_sample:
                sample_id = score.metadata.get("sample_id")
                if sample_id is not None:
                    score_lookup[str(sample_id)] = score.value

        for record in records:
            sample_id = str(record.task.metadata.get("dataset_id"))
            subject = str(record.task.metadata.get(self._subject_field, "unknown"))
            value = score_lookup.get(sample_id)
            if value is not None:
                subject_scores[subject].append(value)

        subject_metrics: list[core_entities.MetricScore] = []
        for subject, values in subject_scores.items():
            avg = mean(values)
            subject_metrics.append(
                core_entities.MetricScore(
                    metric_name="SubjectExactMatch",
                    value=avg,
                    metadata={"subject": subject, "count": len(values)},
                    details={},
                )
            )
        if subject_metrics:
            report.metrics["SubjectExactMatch"] = base_pipeline.MetricAggregate(
                name="SubjectExactMatch",
                count=len(subject_metrics),
                mean=mean(score.value for score in subject_metrics),
                per_sample=subject_metrics,
            )
        self.subject_breakdown = {
            score.metadata["subject"]: score.value for score in subject_metrics
        }
        return report


__all__ = ["SubjectAwareEvaluationPipeline"]
