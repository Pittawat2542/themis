"""Adaptive evaluation pipeline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from themis.core import entities as core_entities
from themis.evaluation import pipeline, reports
from themis.interfaces import Metric
from themis.utils import tracing


class AdaptiveEvaluationPipeline(pipeline.EvaluationPipeline):
    """Pipeline that selects metrics based on sample characteristics.

    This pipeline allows different metrics to be applied to different
    samples based on their metadata, task type, or other characteristics.

    This is more efficient than ConditionalMetric when you have many
    samples that can be grouped by their metric requirements.

    Example:
        >>> def select_metrics(record):
        ...     task_type = record.task.metadata.get("type")
        ...     if task_type == "math":
        ...         return [ExactMatch(), MathVerifyAccuracy()]
        ...     elif task_type == "code":
        ...         return [CodeExecutionMetric()]
        ...     return [ExactMatch()]
        >>>
        >>> pipeline = AdaptiveEvaluationPipeline(
        ...     extractor=extractor,
        ...     metric_selector=select_metrics
        ... )
    """

    def __init__(
        self,
        *,
        extractor: Any,
        metric_selector: Callable[[core_entities.GenerationRecord], list[Metric]],
        **kwargs: Any,
    ):
        """Initialize adaptive pipeline.

        Args:
            extractor: Extractor for all samples
            metric_selector: Function that selects metrics for each record
            **kwargs: Additional arguments passed to EvaluationPipeline
        """
        # Initialize with empty metrics - we'll select them dynamically
        super().__init__(extractor=extractor, metrics=[], **kwargs)
        self._metric_selector = metric_selector

    def evaluate(
        self, records: Sequence[core_entities.GenerationRecord]
    ) -> pipeline.EvaluationReport:
        """Evaluate records with adaptive metric selection.

        Args:
            records: Generation records to evaluate

        Returns:
            Evaluation report
        """
        with tracing.span("adaptive_evaluation", num_records=len(records)):
            # Group records by which metrics apply
            metric_groups: dict[
                tuple[str, ...], list[core_entities.GenerationRecord]
            ] = defaultdict(list)
            record_metrics: dict[str, list[Metric]] = {}

            # Phase 1: Group records by metric set
            with tracing.span("group_by_metrics"):
                for record in records:
                    selected_metrics = self._metric_selector(record)
                    metric_key = tuple(m.name for m in selected_metrics)
                    metric_groups[metric_key].append(record)

                    # Store mapping for later
                    sample_id = str(record.task.metadata.get("dataset_id", "unknown"))
                    record_metrics[sample_id] = selected_metrics

            # Phase 2: Evaluate each group with appropriate metrics
            all_eval_records = []
            all_failures: list[reports.EvaluationFailure] = []
            with tracing.span("evaluate_groups", num_groups=len(metric_groups)):
                for metric_key, group_records in metric_groups.items():
                    if not group_records:
                        continue

                    # Get metrics for this group
                    sample_id = str(
                        group_records[0].task.metadata.get("dataset_id", "unknown")
                    )
                    group_metrics = record_metrics.get(sample_id, [])

                    with tracing.span(
                        "evaluate_group",
                        metric_names=list(metric_key),
                        num_records=len(group_records),
                    ):
                        # Create temporary pipeline for this group
                        temp_pipeline = pipeline.EvaluationPipeline(
                            extractor=self._extractor,
                            metrics=group_metrics,
                        )

                        # Evaluate group
                        group_report = temp_pipeline.evaluate(group_records)
                        all_eval_records.extend(group_report.records)
                        all_failures.extend(group_report.failures)

            # Phase 3: Aggregate all results
            with tracing.span("aggregate_adaptive_results"):
                # Collect all metric scores by metric name
                metric_scores_by_name: dict[str, list[core_entities.MetricScore]] = (
                    defaultdict(list)
                )
                for eval_record in all_eval_records:
                    for score_record in eval_record.scores:
                        metric_scores_by_name[score_record.metric_name].append(
                            score_record
                        )

                # Compute aggregates
                metric_aggregates = {}
                for metric_name, score_objs in metric_scores_by_name.items():
                    if score_objs:
                        metric_aggregates[metric_name] = (
                            reports.MetricAggregate.from_scores(
                                name=metric_name,
                                scores=score_objs,
                            )
                        )

                return reports.EvaluationReport(
                    metrics=metric_aggregates,
                    failures=all_failures,
                    records=all_eval_records,
                )
