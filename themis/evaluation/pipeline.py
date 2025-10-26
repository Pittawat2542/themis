"""Evaluation pipeline orchestration."""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors, strategies as evaluation_strategies
from themis.evaluation.reports import (
    EvaluationFailure,
    EvaluationReport,
    MetricAggregate,
)
from themis.interfaces import Metric as MetricInterface

logger = logging.getLogger(__name__)


def _default_reference_selector(record: core_entities.GenerationRecord):
    reference = record.task.reference
    if reference is None:
        return None
    return reference.value


def _normalize_references(reference):
    if isinstance(reference, core_entities.Reference):
        reference = reference.value
    if isinstance(reference, list):
        return reference
    return [reference]


class EvaluationPipeline:
    def __init__(
        self,
        *,
        extractor,
        metrics: Sequence[MetricInterface],
        reference_selector: Callable[[core_entities.GenerationRecord], object]
        | None = None,
        strategy_resolver: Callable[
            [core_entities.GenerationRecord], evaluation_strategies.EvaluationStrategy
        ]
        | None = None,
    ) -> None:
        self._extractor = extractor
        self._metrics = list(metrics)
        self._reference_selector = reference_selector or _default_reference_selector
        self._strategy_resolver = strategy_resolver or (
            lambda record: evaluation_strategies.DefaultEvaluationStrategy()
        )

    def evaluate(
        self, records: Sequence[core_entities.GenerationRecord]
    ) -> EvaluationReport:
        per_metric: dict[str, list[core_entities.MetricScore]] = {
            metric.name: [] for metric in self._metrics
        }
        failures: list[EvaluationFailure] = []
        per_record: list[core_entities.EvaluationRecord] = []

        for record in records:
            logger.debug(
                "Evaluating sample %s with %s metric(s)",
                record.task.metadata.get("dataset_id")
                or record.task.metadata.get("sample_id"),
                len(self._metrics),
            )
            strategy = self._strategy_resolver(record)
            task_metadata = record.task.metadata
            sample_id = task_metadata.get("dataset_id") or task_metadata.get(
                "sample_id"
            )
            eval_items = list(strategy.prepare(record))
            item_scores: list[core_entities.MetricScore] = []
            record_failures: list[str] = []

            for item in eval_items:
                if item.record.output is None:
                    message = "Missing model output"
                    failures.append(
                        EvaluationFailure(sample_id=sample_id, message=message)
                    )
                    record_failures.append(message)
                    continue
                try:
                    prediction = self._extractor.extract(item.record.output.text)
                except extractors.FieldExtractionError as exc:
                    message = str(exc)
                    failures.append(
                        EvaluationFailure(sample_id=sample_id, message=message)
                    )
                    record_failures.append(message)
                    continue

                reference = item.reference or self._reference_selector(record)
                if reference is None:
                    continue
                references = _normalize_references(reference)
                metadata = {"sample_id": sample_id}
                extract_start = time.perf_counter()
                for metric in self._metrics:
                    metric_start = time.perf_counter()
                    try:
                        score = metric.compute(
                            prediction=prediction,
                            references=references,
                            metadata=metadata,
                        )
                        score.metadata["evaluation_time_ms"] = (
                            time.perf_counter() - metric_start
                        ) * 1000
                    except Exception as exc:  # pragma: no cover - guarded
                        message = (
                            f"Metric '{metric.name}' failed for sample {sample_id}: {exc}"
                        )
                        logger.warning(message)
                        failures.append(
                            EvaluationFailure(sample_id=sample_id, message=message)
                        )
                        record_failures.append(message)
                        score = core_entities.MetricScore(
                            metric_name=metric.name,
                            value=0.0,
                            details={"error": str(exc), "skipped": True},
                            metadata=dict(metadata),
                        )
                        score.metadata["evaluation_time_ms"] = 0.0
                    item_scores.append(score)
                extraction_duration = (time.perf_counter() - extract_start) * 1000
                for score in item_scores[-len(self._metrics) :]:
                    score.metadata.setdefault(
                        "extraction_time_ms", extraction_duration
                    )

            if record_failures and not item_scores:
                metadata = {"sample_id": sample_id}
                for metric in self._metrics:
                    item_scores.append(
                        core_entities.MetricScore(
                            metric_name=metric.name,
                            value=0.0,
                            details={"skipped": True},
                            metadata=metadata,
                        )
                    )

            aggregated_scores = strategy.aggregate(record, item_scores)
            for score in aggregated_scores:
                per_metric[score.metric_name].append(score)
            per_record.append(
                core_entities.EvaluationRecord(
                    sample_id=sample_id,
                    scores=aggregated_scores,
                    failures=record_failures,
                )
            )

        aggregates = {
            name: MetricAggregate.from_scores(name, scores)
            for name, scores in per_metric.items()
        }

        return EvaluationReport(
            metrics=aggregates, failures=failures, records=per_record
        )
