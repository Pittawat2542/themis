"""Composable evaluation pipeline with chainable steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from themis.core import entities as core_entities
from themis.interfaces import Metric as MetricInterface
from themis.utils import tracing

# Type variables for composable pipeline
T = TypeVar("T")
U = TypeVar("U")


@dataclass
class EvaluationStep(Generic[T, U]):
    """Single step in evaluation pipeline.

    A step transforms an input of type T to output of type U.
    It can optionally handle errors that occur during processing.

    Attributes:
        name: Step name
        processor: Function to transform input to output
        error_handler: Optional error handler
    """

    name: str
    processor: Callable[[T], U]
    error_handler: Callable[[Exception], U | None] | None = None

    def execute(self, value: T) -> tuple[U | None, str | None]:
        """Execute the step.

        Args:
            value: Input value

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
        """Check if evaluation succeeded.

        Returns:
            True if no errors and has scores
        """
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
        """Initialize empty pipeline."""
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
        extractor: Any,
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

                        if result is not None:
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
