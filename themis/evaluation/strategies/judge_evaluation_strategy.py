from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from themis.core import entities as core_entities


@dataclass
class JudgeEvaluationStrategy:
    """Aggregate multiple judge metric scores and report agreement.

    This strategy groups incoming MetricScore items by metric_name and returns
    a single aggregated score per metric, including inter-judge agreement.
    It is model-agnostic and works with RubricJudgeMetric and PairwiseJudgeMetric.
    """

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
        if not scores:
            return []
        grouped: dict[str, list[core_entities.MetricScore]] = {}
        for score in scores:
            grouped.setdefault(score.metric_name, []).append(score)

        aggregated: list[core_entities.MetricScore] = []
        for metric_name, group in grouped.items():
            value = sum(item.value for item in group) / max(1, len(group))
            labels: list[str] = []
            for item in group:
                details = item.details or {}
                label = details.get("verdict") or details.get("preference")
                if isinstance(label, str) and label:
                    labels.append(label.lower().strip())
            agreement = 0.0
            if labels:
                from collections import Counter

                counts = Counter(labels)
                agreement = max(counts.values()) / max(1, len(labels))

            # Preserve original metadata from first score
            base_metadata = group[0].metadata.copy() if group[0].metadata else {}
            aggregated.append(
                core_entities.MetricScore(
                    metric_name=metric_name,
                    value=value,
                    details={
                        "judge_count": len(group),
                        "agreement": agreement,
                        "labels": labels,
                    },
                    metadata={
                        **base_metadata,  # Preserve all original metadata
                        "sample_id": base_metadata.get("sample_id"),
                    },
                )
            )
        return aggregated
