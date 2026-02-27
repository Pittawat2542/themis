"""Shared dataclasses that represent Themis' internal world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from themis.evaluation.reports import EvaluationReport

# Type variable for generic Reference
T = TypeVar("T")


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    provider: str
    default_sampling: SamplingConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_key(self) -> str:
        return f"{self.provider}:{self.identifier}"


@dataclass(frozen=True)
class PromptSpec:
    name: str
    template: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptRender:
    spec: PromptSpec
    text: str
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return self.text

    @property
    def template_name(self) -> str:
        return self.spec.name


@dataclass(frozen=True)
class Reference(Generic[T]):
    """Reference value with optional type information.

    This is a generic dataclass that can hold typed reference values.
    It can also be used without type parameters and will behave like
    Reference[Any].

    The value field can hold any type including:
    - Simple types: str, int, float, bool
    - Collections: list, tuple, set
    - Dictionaries: dict (for multi-value references)
    - Custom objects

    Examples:
        # Simple reference
        ref = Reference(kind="answer", value="42")

        # Multi-value reference using dict
        ref = Reference(
            kind="countdown_task",
            value={"target": 122, "numbers": [25, 50, 75, 100]}
        )

        # List reference
        ref = Reference(kind="valid_answers", value=["yes", "no", "maybe"])

        # Typed reference
        ref: Reference[str] = Reference(kind="answer", value="42")
        ref: Reference[dict] = Reference(kind="task", value={"a": 1, "b": 2})

    Note:
        When using dict values, metrics can access individual fields directly:
        >>> target = reference.value["target"]
        >>> numbers = reference.value["numbers"]
    """

    kind: str
    value: T
    schema: type[T] | None = None  # Optional runtime type information


@dataclass(frozen=True)
class ModelOutput:
    text: str
    raw: Any | None = None
    usage: dict[str, int] | None = (
        None  # Token usage: {prompt_tokens, completion_tokens, total_tokens}
    )


@dataclass(frozen=True)
class ModelError:
    message: str
    kind: str = "model_error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationTask:
    prompt: PromptRender
    model: ModelSpec
    sampling: SamplingConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    reference: Reference | None = None


@dataclass
class GenerationRecord:
    task: GenerationTask
    output: ModelOutput | None
    error: ModelError | None
    metrics: dict[str, Any] = field(default_factory=dict)
    attempts: list["GenerationRecord"] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationItem:
    record: GenerationRecord
    reference: Reference | None


@dataclass(frozen=True)
class MetricScore:
    metric_name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    scores: list[MetricScore]
    failures: list[str] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    sample_id: str | None
    scores: list[MetricScore]
    failures: list[str] = field(default_factory=list)


@dataclass
class ExperimentFailure:
    sample_id: str | None
    message: str


@dataclass
class ExperimentReport:
    generation_results: list[GenerationRecord]
    evaluation_report: "EvaluationReport"
    failures: list[ExperimentFailure]
    metadata: dict[str, object]

    def metric(self, name: str) -> Any:
        """Get a metric aggregate by name (case-insensitive, supports snake_case).

        This is a convenience accessor that normalises the lookup key so
        both ``"exact_match"`` and ``"ExactMatch"`` resolve correctly.

        Args:
            name: Metric name — accepts snake_case, CamelCase, or exact key.

        Returns:
            The ``MetricAggregate`` for the requested metric.

        Raises:
            KeyError: If no metric matches *name*.

        Example:
            >>> agg = report.metric("exact_match")
            >>> print(f"{agg.mean:.2%}")
        """
        import re

        metrics = self.evaluation_report.metrics
        # 1. Exact key match
        if name in metrics:
            return metrics[name]
        # 2. Normalised comparison: convert both sides to snake_case lower
        name_norm = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        for key, value in metrics.items():
            key_norm = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
            if key_norm == name_norm:
                return value
        available = ", ".join(sorted(metrics.keys()))
        from themis.exceptions import MetricError

        raise MetricError(f"Metric '{name}' not found. Available: {available}")


__all__ = [
    "SamplingConfig",
    "ModelSpec",
    "PromptSpec",
    "PromptRender",
    "Reference",
    "ModelOutput",
    "ModelError",
    "GenerationTask",
    "GenerationRecord",
    "EvaluationItem",
    "EvaluationRecord",
    "MetricScore",
    "EvaluationSummary",
    "ExperimentFailure",
    "ExperimentReport",
]
