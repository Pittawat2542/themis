"""Shared dataclasses that represent Themis' internal world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, TypeVar

if TYPE_CHECKING:
    from themis.evaluation.reports import EvaluationReport

# Type variable for generic Reference
T = TypeVar("T")


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    provider: str
    default_sampling: SamplingConfig | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSpec:
    name: str
    template: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptRender:
    spec: PromptSpec
    text: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    For backward compatibility, it can be used without type parameters
    and will behave like Reference[Any].

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
    usage: Dict[str, int] | None = None  # Token usage: {prompt_tokens, completion_tokens, total_tokens}


@dataclass(frozen=True)
class ModelError:
    message: str
    kind: str = "model_error"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationTask:
    prompt: PromptRender
    model: ModelSpec
    sampling: SamplingConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference: Reference | None = None


@dataclass
class GenerationRecord:
    task: GenerationTask
    output: ModelOutput | None
    error: ModelError | None
    metrics: Dict[str, Any] = field(default_factory=dict)
    attempts: List["GenerationRecord"] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationItem:
    record: GenerationRecord
    reference: Reference | None


@dataclass(frozen=True)
class MetricScore:
    metric_name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    scores: List[MetricScore]
    failures: List[str] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    sample_id: str | None
    scores: List[MetricScore]
    failures: List[str] = field(default_factory=list)


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
