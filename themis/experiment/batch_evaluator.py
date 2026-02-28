from __future__ import annotations

import logging

from themis.core.entities import (
    EvaluationRecord,
    MetricScore,
)
from themis.evaluation import pipeline as evaluation_pipeline
from themis.evaluation.reports import EvaluationFailure
from themis.experiment.cache_manager import CacheManager
from themis.experiment.context import _ExperimentContext, _RetentionBuffer
import re

logger = logging.getLogger(__name__)


def _stable_metric_id(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", normalized).lower()


class BatchEvaluator:
    """Manages batching, evaluating, and aggregating generation records."""

    def __init__(
        self,
        evaluation_pipeline: evaluation_pipeline.EvaluationPipeline,
        cache_manager: CacheManager,
    ) -> None:
        self._evaluation = evaluation_pipeline
        self._cache = cache_manager

    def evaluate_batch(self, ctx: _ExperimentContext) -> None:
        """Evaluate keyed batch of records and update context."""
        if not ctx.eval_batch:
            return

        logger.info(
            "BatchEvaluator: Evaluating batch of %s records...",
            len(ctx.eval_batch),
            extra={"batch_size": len(ctx.eval_batch)},
        )

        batch_report = self._evaluation.evaluate(ctx.eval_batch)
        ctx.expected_metric_names.update(batch_report.metrics.keys())

        for record, evaluation in zip(ctx.eval_batch, batch_report.records):
            if ctx.cache_results:
                self._cache.save_evaluation_record(
                    ctx.run_identifier, record, evaluation, ctx.evaluation_config
                )
            self.update_metric_stats(ctx, evaluation)
            ctx.new_eval_records.append(evaluation)

        ctx.new_eval_failures.extend(batch_report.failures)
        ctx.eval_batch = []

    def update_metric_stats(
        self, ctx: _ExperimentContext, record: EvaluationRecord
    ) -> None:
        """Update metric statistics from a single evaluation record."""
        ctx.evaluation_record_failures_total += len(record.failures)
        for score in record.scores:
            metric_name = score.metric_name
            ctx.metric_sums[metric_name] = (
                ctx.metric_sums.get(metric_name, 0.0) + score.value
            )
            ctx.metric_counts[metric_name] = ctx.metric_counts.get(metric_name, 0) + 1
            buffer = ctx.metric_samples.setdefault(
                metric_name, _RetentionBuffer(ctx.generation_results.max_items)
            )
            buffer.append(score)

    def combine_evaluations(
        self,
        cached_records: list[EvaluationRecord],
        new_records: list[EvaluationRecord],
        new_failures: list[EvaluationFailure],
        *,
        expected_metric_names: set[str] | None = None,
        cached_failures: list[EvaluationFailure] | None = None,
        metric_sums: dict[str, float] | None = None,
        metric_counts: dict[str, int] | None = None,
        metric_samples: dict[str, list[MetricScore]] | None = None,
        max_records_in_memory: int | None = None,
    ) -> evaluation_pipeline.EvaluationReport:
        record_buffer = _RetentionBuffer(max_records_in_memory)
        for record in cached_records:
            record_buffer.append(record)
        for record in new_records:
            record_buffer.append(record)
        all_records = record_buffer.to_list()

        if metric_sums is None or metric_counts is None or metric_samples is None:
            per_metric: dict[str, list[MetricScore]] = {}
            computed_sums: dict[str, float] = {}
            computed_counts: dict[str, int] = {}
            for record in all_records:
                for score in record.scores:
                    per_metric.setdefault(score.metric_name, []).append(score)
                    computed_sums[score.metric_name] = (
                        computed_sums.get(score.metric_name, 0.0) + score.value
                    )
                    computed_counts[score.metric_name] = (
                        computed_counts.get(score.metric_name, 0) + 1
                    )
            metric_sums = computed_sums
            metric_counts = computed_counts
            metric_samples = per_metric

        aggregates: dict[str, evaluation_pipeline.MetricAggregate] = {}
        metric_names = set(metric_counts.keys()) | set(expected_metric_names or set())
        metrics_truncated = False
        per_metric_truncation: dict[str, dict[str, int | bool]] = {}
        for name in metric_names:
            scores = metric_samples.get(name, [])
            count = metric_counts.get(name, 0)
            mean = (metric_sums.get(name, 0.0) / count) if count else 0.0
            truncated_count = max(0, count - len(scores))
            per_sample_complete = truncated_count == 0
            metrics_truncated = metrics_truncated or (not per_sample_complete)
            per_metric_truncation[name] = {
                "per_sample_complete": per_sample_complete,
                "truncated_count": truncated_count,
            }
            aggregates[name] = evaluation_pipeline.MetricAggregate(
                name=name,
                count=count,
                mean=mean,
                per_sample=list(scores),
                per_sample_complete=per_sample_complete,
                truncated_count=truncated_count,
            )

        failures = list(new_failures)
        if cached_failures is not None:
            failures.extend(cached_failures)
        else:
            for record in cached_records:
                for message in record.failures:
                    failures.append(
                        EvaluationFailure(sample_id=record.sample_id, message=message)
                    )

        return evaluation_pipeline.EvaluationReport(
            metrics=aggregates,
            failures=failures,
            records=all_records,
            metadata={
                "max_records_in_memory": max_records_in_memory,
                "records_retained": len(all_records),
                "records_dropped": record_buffer.dropped,
                "per_sample_metrics_truncated": metrics_truncated,
                "per_metric": per_metric_truncation,
                "metric_ids": {
                    metric_name: _stable_metric_id(metric_name)
                    for metric_name in metric_names
                },
            },
        )
