from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from themis.core import entities as core_entities


@dataclass
class AttemptAwareEvaluationStrategy:
    """Evaluates each generation attempt independently.

    When average_attempts=True, returns a single averaged score per metric.
    """

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
        aggregated: list[core_entities.MetricScore] = []
        grouped: dict[str, list[core_entities.MetricScore]] = {}
        for score in scores:
            grouped.setdefault(score.metric_name, []).append(score)
        for metric_name, group in grouped.items():
            value = sum(item.value for item in group) / len(group)
            # Preserve original metadata from first score
            base_metadata = group[0].metadata.copy() if group[0].metadata else {}
            aggregated.append(
                core_entities.MetricScore(
                    metric_name=metric_name,
                    value=value,
                    metadata={
                        **base_metadata,  # Preserve all original metadata
                        "attempts": len(group),  # Add aggregation-specific field
                        "sample_id": base_metadata.get("sample_id"),
                    },
                    details={},
                )
            )
        return aggregated
