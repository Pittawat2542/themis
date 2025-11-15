"""Evaluation pipeline orchestration.

This module provides two complementary pipeline styles:

1. EvaluationPipeline: Traditional batch evaluation with extractors, metrics, and strategies
2. ComposableEvaluationPipeline: Chainable builder pattern for composing evaluation steps

Example (Traditional):
    >>> pipeline = EvaluationPipeline(
    ...     extractor=JsonFieldExtractor("answer"),
    ...     metrics=[ExactMatch()]
    ... )
    >>> report = pipeline.evaluate(records)

Example (Composable):
    >>> pipeline = (
    ...     ComposableEvaluationPipeline()
    ...     .extract(JsonFieldExtractor("answer"))
    ...     .validate(lambda x: isinstance(x, str), "Must be string")
    ...     .transform(lambda x: x.strip().lower(), name="normalize")
    ...     .compute_metrics([ExactMatch()], references=["42"])
    ... )
    >>> result = pipeline.evaluate(record)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from themis.core import entities as core_entities
from themis.evaluation import extractors, strategies as evaluation_strategies
from themis.evaluation.reports import (
    EvaluationFailure,
    EvaluationReport,
    MetricAggregate,
)
from themis.interfaces import Metric as MetricInterface
from themis.utils import tracing

logger = logging.getLogger(__name__)

# Type variables for composable pipeline
T = TypeVar("T")
U = TypeVar("U")


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
        with tracing.span("evaluate_pipeline", total_records=len(records)):
            per_metric: dict[str, list[core_entities.MetricScore]] = {
                metric.name: [] for metric in self._metrics
            }
            failures: list[EvaluationFailure] = []
            per_record: list[core_entities.EvaluationRecord] = []

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

                        reference = item.reference or self._reference_selector(record)
                        if reference is None:
                            continue
                        references = _normalize_references(reference)
                        metadata = {"sample_id": sample_id}
                        extract_start = time.perf_counter()
                        for metric in self._metrics:
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
                            except Exception as exc:  # pragma: no cover - guarded
                                message = f"Metric '{metric.name}' failed for sample {sample_id}: {exc}"
                                logger.warning(message)
                                failures.append(
                                    EvaluationFailure(
                                        sample_id=sample_id, message=message
                                    )
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
                        extraction_duration = (
                            time.perf_counter() - extract_start
                        ) * 1000
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


# ============================================================================
# Composable Evaluation Pipeline
# ============================================================================


@dataclass
class EvaluationStep(Generic[T, U]):
    """Single step in evaluation pipeline.

    A step transforms an input of type T to output of type U.
    It can optionally handle errors that occur during processing.
    """

    name: str
    processor: Callable[[T], U]
    error_handler: Callable[[Exception], U | None] | None = None

    def execute(self, value: T) -> tuple[U | None, str | None]:
        """Execute the step.

        Returns:
            Tuple of (result, error_message)
        """
        try:
            result = self.processor(value)
            return result, None
        except Exception as e:
            if self.error_handler:
                handled = self.error_handler(e)
                if handled is not None:
                    return handled, None
            return None, str(e)


@dataclass
class EvaluationResult:
    """Result from evaluating a single record through pipeline.

    Attributes:
        record: Original generation record
        scores: Final metric scores
        errors: List of errors encountered
        intermediate_values: Dict of intermediate values from each step
    """

    record: core_entities.GenerationRecord
    scores: list[core_entities.MetricScore]
    errors: list[str]
    intermediate_values: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if evaluation succeeded."""
        return len(self.errors) == 0 and len(self.scores) > 0


class ComposableEvaluationPipeline:
    """Pipeline that chains multiple evaluation steps.

    This pipeline allows you to compose evaluation logic from multiple steps:
    1. Extraction (get answer from raw output)
    2. Validation (check format/constraints)
    3. Transformation (normalize, clean, convert)
    4. Metric computation (compare against references)

    Each step can have error handling, and intermediate values are tracked.

    Example:
        >>> pipeline = (
        ...     ComposableEvaluationPipeline()
        ...     .extract(RegexExtractor(r"(\\d+)"))
        ...     .validate(lambda x: x.isdigit(), "Must be numeric")
        ...     .transform(int, name="parse_int")
        ...     .compute_metrics([NumericMatch()], references=[42])
        ... )
    """

    def __init__(self):
        self._steps: list[EvaluationStep] = []

    def add_step(self, step: EvaluationStep) -> ComposableEvaluationPipeline:
        """Add a step to the pipeline (builder pattern).

        Args:
            step: Evaluation step to add

        Returns:
            Self for chaining
        """
        self._steps.append(step)
        return self

    def extract(
        self,
        extractor: extractors.Extractor,
        error_handler: Callable[[Exception], Any | None] | None = None,
    ) -> ComposableEvaluationPipeline:
        """Add extraction step.

        Args:
            extractor: Extractor to use
            error_handler: Optional error handler

        Returns:
            Self for chaining
        """
        return self.add_step(
            EvaluationStep(
                name=f"extract_{extractor.__class__.__name__}",
                processor=extractor.extract,
                error_handler=error_handler,
            )
        )

    def validate(
        self, validator: Callable[[Any], bool], error_message: str = "Validation failed"
    ) -> ComposableEvaluationPipeline:
        """Add validation step.

        Args:
            validator: Function that returns True if valid
            error_message: Error message if validation fails

        Returns:
            Self for chaining
        """

        def validate_fn(value):
            if not validator(value):
                raise ValueError(error_message)
            return value

        return self.add_step(
            EvaluationStep(
                name="validate",
                processor=validate_fn,
            )
        )

    def transform(
        self,
        transformer: Callable[[Any], Any],
        name: str = "transform",
        error_handler: Callable | None = None,
    ) -> ComposableEvaluationPipeline:
        """Add transformation step.

        Args:
            transformer: Function to transform value
            name: Name for this step
            error_handler: Optional error handler

        Returns:
            Self for chaining
        """
        return self.add_step(
            EvaluationStep(
                name=name,
                processor=transformer,
                error_handler=error_handler,
            )
        )

    def conditional_step(
        self,
        condition: Callable[[Any], bool],
        step_if_true: EvaluationStep,
        step_if_false: EvaluationStep | None = None,
    ) -> ComposableEvaluationPipeline:
        """Add conditional step that branches based on condition.

        Args:
            condition: Function to determine which branch to take
            step_if_true: Step to execute if condition is True
            step_if_false: Step to execute if condition is False (or passthrough)

        Returns:
            Self for chaining
        """

        def conditional_processor(value):
            if condition(value):
                result, error = step_if_true.execute(value)
                if error:
                    raise ValueError(f"True branch failed: {error}")
                return result
            elif step_if_false:
                result, error = step_if_false.execute(value)
                if error:
                    raise ValueError(f"False branch failed: {error}")
                return result
            else:
                return value  # Passthrough

        return self.add_step(
            EvaluationStep(
                name=f"conditional_{step_if_true.name}",
                processor=conditional_processor,
            )
        )

    def compute_metrics(
        self,
        metrics: Sequence[MetricInterface],
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> ComposableEvaluationPipeline:
        """Add metrics computation step.

        This should typically be the final step in the pipeline.

        Args:
            metrics: List of metrics to compute
            references: Reference values to compare against
            metadata: Optional metadata to pass to metrics

        Returns:
            Self for chaining
        """

        def compute(prediction):
            scores = []
            for metric in metrics:
                score = metric.compute(
                    prediction=prediction,
                    references=references,
                    metadata=metadata or {},
                )
                scores.append(score)
            return scores

        return self.add_step(
            EvaluationStep(
                name="compute_metrics",
                processor=compute,
            )
        )

    def evaluate(self, record: core_entities.GenerationRecord) -> EvaluationResult:
        """Execute the pipeline on a generation record.

        Args:
            record: Generation record to evaluate

        Returns:
            Evaluation result with scores, errors, and intermediate values
        """
        if record.output is None:
            return EvaluationResult(
                record=record,
                scores=[],
                errors=["Missing model output"],
                intermediate_values={},
            )

        intermediate_values = {"raw_output": record.output.text}
        current_value = record.output.text
        errors = []

        with tracing.span("composable_pipeline_evaluate", num_steps=len(self._steps)):
            for step in self._steps:
                try:
                    with tracing.span(f"eval_step_{step.name}"):
                        result, error = step.execute(current_value)

                        if error:
                            errors.append(f"{step.name}: {error}")
                            return EvaluationResult(
                                record=record,
                                scores=[],
                                errors=errors,
                                intermediate_values=intermediate_values,
                            )

                        current_value = result
                        intermediate_values[step.name] = current_value

                except Exception as e:
                    errors.append(f"{step.name}: {str(e)}")
                    return EvaluationResult(
                        record=record,
                        scores=[],
                        errors=errors,
                        intermediate_values=intermediate_values,
                    )

        # Final value should be list of scores if compute_metrics was last step
        scores = current_value if isinstance(current_value, list) else []

        # Filter to only MetricScore objects
        metric_scores = [s for s in scores if isinstance(s, core_entities.MetricScore)]

        return EvaluationResult(
            record=record,
            scores=metric_scores,
            errors=errors,
            intermediate_values=intermediate_values,
        )

    def evaluate_batch(
        self, records: Sequence[core_entities.GenerationRecord]
    ) -> list[EvaluationResult]:
        """Evaluate multiple records.

        Args:
            records: List of generation records

        Returns:
            List of evaluation results
        """
        results = []
        with tracing.span("composable_pipeline_batch", num_records=len(records)):
            for record in records:
                result = self.evaluate(record)
                results.append(result)
        return results

    def get_step_names(self) -> list[str]:
        """Get names of all steps in pipeline.

        Returns:
            List of step names
        """
        return [step.name for step in self._steps]

    def clear(self) -> ComposableEvaluationPipeline:
        """Clear all steps from pipeline.

        Returns:
            Self for chaining
        """
        self._steps.clear()
        return self
