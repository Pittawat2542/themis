"""Conditional and adaptive evaluation strategies.

This module provides evaluation components that adapt based on sample characteristics:
- ConditionalMetric: Only runs when condition is met
- AdaptiveEvaluationPipeline: Selects metrics based on sample metadata
- Metric selectors: Helper functions for common selection patterns

Example:
    >>> # Only run math verification on math problems
    >>> math_metric = ConditionalMetric(
    ...     metric=MathVerifyAccuracy(),
    ...     condition=lambda record: record.task.metadata.get("type") == "math"
    ... )
    >>>
    >>> # Adaptively select metrics based on task type
    >>> def select_metrics(record):
    ...     if record.task.metadata.get("type") == "math":
    ...         return [ExactMatch(), MathVerifyAccuracy()]
    ...     return [ExactMatch()]
    >>>
    >>> pipeline = AdaptiveEvaluationPipeline(
    ...     extractor=extractor,
    ...     metric_selector=select_metrics
    ... )
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from themis.core import entities as core_entities
from themis.evaluation import pipeline, reports
from themis.interfaces import Metric
from themis.utils import tracing


@dataclass
class ConditionalMetric:
    """Metric that only runs when condition is met.

    This wrapper allows you to conditionally apply metrics based on
    record characteristics (metadata, task type, etc.).

    Attributes:
        metric: Wrapped metric
        condition: Function that determines if metric should run
        default_score: Score to return when condition is False
        name: Optional override for metric name

    Example:
        >>> # Only run expensive metric on hard problems
        >>> hard_metric = ConditionalMetric(
        ...     metric=ExpensiveVerification(),
        ...     condition=lambda r: r.task.metadata.get("difficulty") == "hard",
        ...     default_score=0.0
        ... )
    """

    metric: Metric
    condition: Callable[[core_entities.GenerationRecord], bool]
    default_score: float = 0.0
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"conditional_{self.metric.name}"

    def should_evaluate(self, record: core_entities.GenerationRecord) -> bool:
        """Check if metric should be evaluated for this record.

        Args:
            record: Generation record

        Returns:
            True if condition is met
        """
        try:
            return self.condition(record)
        except Exception:
            # If condition check fails, don't run metric
            return False

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        """Compute metric score.

        Note: This method doesn't check the condition - it's assumed
        the condition was already checked before calling compute.

        Args:
            prediction: Predicted value
            references: Reference values
            metadata: Optional metadata

        Returns:
            Metric score
        """
        return self.metric.compute(
            prediction=prediction,
            references=references,
            metadata=metadata,
        )

    def compute_or_default(
        self,
        record: core_entities.GenerationRecord,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        """Compute metric or return default score if condition not met.

        Args:
            record: Generation record (for condition check)
            prediction: Predicted value
            references: Reference values
            metadata: Optional metadata

        Returns:
            Metric score or default
        """
        if self.should_evaluate(record):
            return self.compute(
                prediction=prediction,
                references=references,
                metadata=metadata,
            )
        else:
            return core_entities.MetricScore(
                metric_name=self.name or self.metric.name,
                value=self.default_score,
                metadata={"skipped": True, "reason": "condition_not_met"},
            )


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
                    failures=[],  # No failures tracked in adaptive pipeline
                    records=all_eval_records,
                )


# Helper functions for common metric selection patterns


def select_by_metadata_field(
    field: str, metric_map: dict[Any, list[Metric]], default: list[Metric] | None = None
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Create selector that chooses metrics based on metadata field value.

    Args:
        field: Metadata field to check
        metric_map: Mapping from field value to metrics
        default: Default metrics if field value not in map

    Returns:
        Metric selector function

    Example:
        >>> selector = select_by_metadata_field(
        ...     "type",
        ...     {
        ...         "math": [ExactMatch(), MathVerifyAccuracy()],
        ...         "code": [CodeExecutionMetric()],
        ...     },
        ...     default=[ExactMatch()]
        ... )
    """
    default_metrics = default or []

    def selector(record: core_entities.GenerationRecord) -> list[Metric]:
        value = record.task.metadata.get(field)
        return metric_map.get(value, default_metrics)

    return selector


def select_by_difficulty(
    easy_metrics: list[Metric],
    medium_metrics: list[Metric],
    hard_metrics: list[Metric],
    difficulty_field: str = "difficulty",
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Create selector that chooses metrics based on difficulty.

    Args:
        easy_metrics: Metrics for easy problems
        medium_metrics: Metrics for medium problems
        hard_metrics: Metrics for hard problems
        difficulty_field: Name of difficulty field in metadata

    Returns:
        Metric selector function

    Example:
        >>> selector = select_by_difficulty(
        ...     easy_metrics=[ExactMatch()],
        ...     medium_metrics=[ExactMatch(), PartialCredit()],
        ...     hard_metrics=[ExactMatch(), PartialCredit(), ManualReview()]
        ... )
    """
    return select_by_metadata_field(
        difficulty_field,
        {
            "easy": easy_metrics,
            "medium": medium_metrics,
            "hard": hard_metrics,
        },
        default=medium_metrics,
    )


def select_by_condition(
    condition: Callable[[core_entities.GenerationRecord], bool],
    metrics_if_true: list[Metric],
    metrics_if_false: list[Metric],
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Create selector based on arbitrary condition.

    Args:
        condition: Function to determine which metrics to use
        metrics_if_true: Metrics if condition is True
        metrics_if_false: Metrics if condition is False

    Returns:
        Metric selector function

    Example:
        >>> selector = select_by_condition(
        ...     lambda r: len(r.output.text) > 1000,
        ...     metrics_if_true=[SummaryMetrics()],
        ...     metrics_if_false=[ExactMatch()]
        ... )
    """

    def selector(record: core_entities.GenerationRecord) -> list[Metric]:
        try:
            if condition(record):
                return metrics_if_true
            else:
                return metrics_if_false
        except Exception:
            # If condition fails, use false branch
            return metrics_if_false

    return selector


def combine_selectors(
    *selectors: Callable[[core_entities.GenerationRecord], list[Metric]],
) -> Callable[[core_entities.GenerationRecord], list[Metric]]:
    """Combine multiple selectors (union of their metrics).

    Args:
        *selectors: Metric selectors to combine

    Returns:
        Combined selector that returns union of all selected metrics

    Example:
        >>> selector = combine_selectors(
        ...     select_by_type,
        ...     select_by_difficulty,
        ... )
    """

    def combined(record: core_entities.GenerationRecord) -> list[Metric]:
        all_metrics = []
        seen_names = set()

        for selector in selectors:
            selected = selector(record)
            for metric in selected:
                if metric.name not in seen_names:
                    all_metrics.append(metric)
                    seen_names.add(metric.name)

        return all_metrics

    return combined
