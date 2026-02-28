"""Standard evaluation pipeline implementation."""

from __future__ import annotations

import logging
import re
import time
import warnings
from collections.abc import Callable, Sequence

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


def _stable_metric_id(name: str) -> str:
    """Normalize display-like metric names to snake_case IDs."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", normalized).lower()
    return normalized


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
    """The central engine for scoring LLM outputs against references.

    The EvaluationPipeline takes completed `GenerationRecord`s (which contain
    the model's raw text response) and pipes them through:
    1. An Extractor (to pull out the specific answer, like from JSON or XML tags).
    2. A Strategy (to handle single answers vs. multiple samples like Pass@K).
    3. A list of Metrics (to score the extracted answer against the gold reference).

    Example (Standalone Usage):
        ```python
        from themis.evaluation.pipelines import EvaluationPipeline
        from themis.evaluation.extractors import RegexExtractor
        from themis.evaluation.metrics import ExactMatch

        pipeline = EvaluationPipeline(
            extractor=RegexExtractor(pattern=r"<answer>(.*?)</answer>"),
            metrics=[ExactMatch()]
        )
        report = pipeline.evaluate(my_records)
        print(report.metrics["ExactMatch"].mean)
        ```
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
        """Initialize the evaluation pipeline.

        Args:
            extractor: The component responsible for parsing the final answer out of
                the model's raw text.
            metrics: A list of instantiated metrics (e.g., `[ExactMatch(), RougeL()]`)
                to compute against the extracted answer.
            reference_selector: An optional custom function to extract the expected
                reference from the dataset row. If provided, overrides default behavior.
            strategy_resolver: Optional function to dictate how multiple samples
                per prompt are handled (e.g., majority voting, best-of-N).

        Note:
            If you provide a custom `reference_selector` but use the default strategy,
            your custom selector still takes precedence.
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
                        item_start = time.perf_counter()
                        try:
                            extraction_start = time.perf_counter()
                            with tracing.span("extract"):
                                prediction = self._extractor.extract(
                                    item.record.output.text
                                )
                            extraction_duration = (
                                time.perf_counter() - extraction_start
                            ) * 1000
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
                                metric_duration = (
                                    time.perf_counter() - metric_start
                                ) * 1000
                                score.metadata["evaluation_time_ms"] = metric_duration
                                score.metadata["metric_compute_time_ms"] = (
                                    metric_duration
                                )
                                item_scores_for_item.append(score)
                            except Exception as exc:  # pragma: no cover - guarded
                                message = f"Metric '{metric.name}' failed for sample {sample_id}: {exc}"
                                logger.warning(message)
                                failures.append(
                                    EvaluationFailure(
                                        sample_id=sample_id, message=message
                                    )
                                )
                                record_failures.append(message)
                        item_duration = (time.perf_counter() - item_start) * 1000
                        for score in item_scores_for_item:
                            score.metadata["extractor_time_ms"] = extraction_duration
                            score.metadata.setdefault(
                                "extraction_time_ms", extraction_duration
                            )
                            score.metadata["item_evaluation_time_ms"] = item_duration
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
                metadata={
                    "metric_ids": {
                        metric_name: _stable_metric_id(metric_name)
                        for metric_name in aggregates
                    }
                },
            )

    def evaluation_fingerprint(self) -> dict:
        """Return a deterministic fingerprint for cache invalidation."""
        config: dict[str, object] = {}
        config["metrics"] = sorted(
            [
                f"{metric.__class__.__module__}.{metric.__class__.__name__}:{metric.name}"
                for metric in self._metrics
            ]
        )
        extractor = self._extractor
        extractor_type = (
            f"{extractor.__class__.__module__}.{extractor.__class__.__name__}"
        )
        config["extractor"] = extractor_type
        if hasattr(extractor, "field_name"):
            config["extractor_field"] = extractor.field_name
        return config

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
