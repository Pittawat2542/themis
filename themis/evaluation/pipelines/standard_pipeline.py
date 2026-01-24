"""Standard evaluation pipeline implementation."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Callable, Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors
from themis.evaluation import strategies as evaluation_strategies
from themis.evaluation.reports import (
    EvaluationFailure,
    EvaluationReport,
    MetricAggregate,
)
from themis.interfaces import Metric as MetricInterface
from themis.utils import tracing

logger = logging.getLogger(__name__)


def _default_reference_selector(record: core_entities.GenerationRecord):
    """Default reference selector from generation record.

    Args:
        record: Generation record

    Returns:
        Reference value or None
    """
    reference = record.task.reference
    if reference is None:
        return None
    return reference.value


def _normalize_references(reference) -> list:
    """Normalize reference to list format for metric consumption.

    This function converts various reference formats into a standardized list
    that metrics can reliably consume. The normalized format is always a list
    where each element represents one reference value.

    Args:
        reference: Reference value in various formats:
            - Reference object: Extracts .value field
            - dict: Kept as-is in a list (for multi-value references)
            - list/tuple: Returned as list
            - scalar: Wrapped in a list

    Returns:
        List of reference values. Each element can be:
        - A scalar value (str, int, float, bool)
        - A dict (for multi-value references like {"target": 122, "numbers": [...]})
        - Any other type from the original reference

    Examples:
        >>> _normalize_references(Reference(kind="answer", value="42"))
        ["42"]

        >>> _normalize_references(Reference(kind="task", value={"target": 122, "numbers": [25, 50]}))
        [{"target": 122, "numbers": [25, 50]}]

        >>> _normalize_references(["yes", "no", "maybe"])
        ["yes", "no", "maybe"]

        >>> _normalize_references("42")
        ["42"]

    Note:
        Metrics receive references in this normalized format and should handle
        both simple values and dict values appropriately.
    """
    if isinstance(reference, core_entities.Reference):
        reference = reference.value
    if isinstance(reference, list):
        return reference
    if isinstance(reference, tuple):
        return list(reference)
    return [reference]


class EvaluationPipeline:
    """Traditional batch evaluation pipeline.

    This pipeline evaluates generation records using extractors, metrics,
    and evaluation strategies. It supports slicing for subset analysis.

    Example:
        >>> pipeline = EvaluationPipeline(
        ...     extractor=JsonFieldExtractor("answer"),
        ...     metrics=[ExactMatch()]
        ... )
        >>> report = pipeline.evaluate(records)

    Attributes:
        _extractor: Extractor for parsing model output
        _metrics: List of metrics to compute
        _reference_selector: Function to extract reference from record
        _strategy_resolver: Function to resolve evaluation strategy
        _slices: List of (name, predicate) tuples for slicing
    """

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
        """Initialize evaluation pipeline.

        Args:
            extractor: Extractor for parsing model output
            metrics: List of metrics to compute
            reference_selector: Optional function to extract reference from record.
                If provided, this takes precedence over item.reference from strategies.
            strategy_resolver: Optional function to resolve evaluation strategy.
                If using a custom reference_selector with DefaultEvaluationStrategy,
                the selector will take precedence.

        Note:
            When using DefaultEvaluationStrategy with a custom reference_selector,
            the reference_selector will override the default behavior. Consider
            using a custom strategy if you need more control over reference selection.
        """
        self._extractor = extractor
        self._metrics = list(metrics)
        self._reference_selector = reference_selector
        self._has_custom_reference_selector = reference_selector is not None
        self._strategy_resolver = strategy_resolver or (
            lambda record: evaluation_strategies.DefaultEvaluationStrategy()
        )
        self._slices: list[
            tuple[str, Callable[[core_entities.GenerationRecord], bool]]
        ] = []

        # Validation: warn if custom reference_selector is used with default strategy
        if self._has_custom_reference_selector and strategy_resolver is None:
            warnings.warn(
                "Custom reference_selector provided without custom strategy_resolver. "
                "The reference_selector will take precedence over DefaultEvaluationStrategy's "
                "reference handling. If you need more control, consider providing a custom "
                "strategy_resolver that sets reference=None in EvaluationItem.",
                UserWarning,
                stacklevel=2,
            )

    def evaluate(
        self, records: Sequence[core_entities.GenerationRecord]
    ) -> EvaluationReport:
        """Evaluate generation records.

        Args:
            records: Generation records to evaluate

        Returns:
            Evaluation report with metrics and failures
        """
        with tracing.span("evaluate_pipeline", total_records=len(records)):
            per_metric: dict[str, list[core_entities.MetricScore]] = {
                metric.name: [] for metric in self._metrics
            }
            failures: list[EvaluationFailure] = []
            per_record: list[core_entities.EvaluationRecord] = []
            slice_members: dict[str, set[str]] = {
                name: set() for name, _ in self._slices
            }

            for record in records:
                with tracing.span("evaluate_record"):
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
                    for name, fn in self._slices:
                        try:
                            if fn(record) and sample_id is not None:
                                slice_members[name].add(sample_id)
                        except Exception:
                            pass
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
                            with tracing.span("extract"):
                                prediction = self._extractor.extract(
                                    item.record.output.text
                                )
                        except extractors.FieldExtractionError as exc:
                            message = str(exc)
                            failures.append(
                                EvaluationFailure(sample_id=sample_id, message=message)
                            )
                            record_failures.append(message)
                            continue

                        # CRITICAL: Always call reference_selector if provided (takes precedence)
                        # This fixes the issue where DefaultEvaluationStrategy's reference
                        # would prevent custom reference_selector from being called
                        if self._has_custom_reference_selector:
                            reference = self._reference_selector(record)
                        elif item.reference is not None:
                            reference = item.reference
                        else:
                            reference = _default_reference_selector(record)

                        references = (
                            _normalize_references(reference)
                            if reference is not None
                            else []
                        )
                        # Preserve all task metadata for metrics, add sample_id
                        metadata = {**record.task.metadata, "sample_id": sample_id}
                        extract_start = time.perf_counter()
                        item_scores_for_item: list[core_entities.MetricScore] = []
                        for metric in self._metrics:
                            requires_reference = getattr(
                                metric, "requires_reference", True
                            )
                            if requires_reference and not references:
                                message = (
                                    f"Missing reference for metric '{metric.name}'"
                                )
                                failures.append(
                                    EvaluationFailure(
                                        sample_id=sample_id, message=message
                                    )
                                )
                                record_failures.append(message)
                                continue
                            metric_start = time.perf_counter()
                            try:
                                with tracing.span(
                                    "compute_metric", metric_name=metric.name
                                ):
                                    score = metric.compute(
                                        prediction=prediction,
                                        references=references,
                                        metadata=metadata,
                                    )
                                score.metadata["evaluation_time_ms"] = (
                                    time.perf_counter() - metric_start
                                ) * 1000
                                item_scores_for_item.append(score)
                            except Exception as exc:  # pragma: no cover - guarded
                                message = (
                                    f"Metric '{metric.name}' failed for sample {sample_id}: {exc}"
                                )
                                logger.warning(message)
                                failures.append(
                                    EvaluationFailure(
                                        sample_id=sample_id, message=message
                                    )
                                )
                                record_failures.append(message)
                        extraction_duration = (
                            time.perf_counter() - extract_start
                        ) * 1000
                        for score in item_scores_for_item:
                            score.metadata.setdefault(
                                "extraction_time_ms", extraction_duration
                            )
                        item_scores.extend(item_scores_for_item)

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
                metrics=aggregates,
                failures=failures,
                records=per_record,
                slices=self._compute_slice_aggregates(per_metric, slice_members),
            )

    def register_slice(
        self, name: str, fn: Callable[[core_entities.GenerationRecord], bool]
    ) -> None:
        """Register a slice for subset analysis.

        Args:
            name: Slice name
            fn: Predicate function to determine slice membership
        """
        self._slices.append((name, fn))

    def _compute_slice_aggregates(
        self,
        per_metric: dict[str, list[core_entities.MetricScore]],
        slice_members: dict[str, set[str]],
    ) -> dict[str, dict[str, MetricAggregate]]:
        """Compute metric aggregates for each slice.

        Args:
            per_metric: Scores by metric name
            slice_members: Sample IDs by slice name

        Returns:
            Nested dict of slice -> metric -> aggregate
        """
        if not slice_members:
            return {}
        slice_aggregates: dict[str, dict[str, MetricAggregate]] = {}
        for name, members in slice_members.items():
            slice_scores_by_metric: dict[str, list[core_entities.MetricScore]] = {}
            for metric_name, scores in per_metric.items():
                filtered = [s for s in scores if s.metadata.get("sample_id") in members]
                slice_scores_by_metric[metric_name] = filtered
            slice_aggregates[name] = {
                metric_name: MetricAggregate.from_scores(metric_name, scores)
                for metric_name, scores in slice_scores_by_metric.items()
            }
        return slice_aggregates
