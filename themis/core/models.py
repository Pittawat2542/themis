"""Core immutable domain models for Themis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from themis.core.base import HashableModel, JSONValue


class MetricDirection(StrEnum):
    """How to interpret movement in a metric value."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


class ScoreOutcome(StrEnum):
    """Case-level reporting outcome derived from a metric contract."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    SCORED = "scored"
    ERROR = "error"


class SeedCapability(StrEnum):
    """Whether a provider call surface can apply a requested seed."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class MetricInterpretation(HashableModel):
    """Stable interpretation contract for one metric component."""

    direction: MetricDirection = MetricDirection.NEUTRAL
    valid_range: tuple[float, float] | None = None
    correctness_threshold: float | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> MetricInterpretation:
        if self.valid_range is not None:
            lower, upper = self.valid_range
            if lower > upper:
                raise ValueError(
                    "Metric valid_range lower bound must not exceed upper bound"
                )
            if self.correctness_threshold is not None and not (
                lower <= self.correctness_threshold <= upper
            ):
                raise ValueError(
                    "Metric correctness_threshold must be inside valid_range"
                )
        if (
            self.direction is MetricDirection.NEUTRAL
            and self.correctness_threshold is not None
        ):
            raise ValueError("Neutral metrics cannot declare a correctness threshold")
        return self


class Case(HashableModel):
    """One dataset case evaluated by the runtime."""

    case_id: str
    input: JSONValue
    expected_output: JSONValue | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Dataset(HashableModel):
    """A collection of cases evaluated together."""

    dataset_id: str
    cases: list[Case] = Field(default_factory=list)
    revision: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Message(HashableModel):
    """One conversation message captured as an artifact."""

    role: str
    content: JSONValue


class TraceStep(HashableModel):
    """One structured step in a generation or evaluation trace."""

    step_name: str
    step_type: str
    input: dict[str, JSONValue] = Field(default_factory=dict)
    output: dict[str, JSONValue] = Field(default_factory=dict)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    timestamp: datetime | None = None


class StreamEvent(HashableModel):
    """One persisted streaming event emitted during generation or judging."""

    event_id: str
    source_stage: str
    event_type: str
    payload: JSONValue
    timestamp: datetime | None = None
    offset_ms: float | None = None
    duration_ms: float | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ProviderTelemetry(HashableModel):
    """Provider-level metadata captured for a generation or judge call."""

    provider_id: str | None = None
    model_id: str | None = None
    latency_ms: float = 0.0
    retry_count: int = 0
    token_usage: dict[str, int] | None = None
    failure_category: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    request_id: str | None = None
    raw_response: dict[str, JSONValue] = Field(default_factory=dict)
    headers: dict[str, JSONValue] | None = None
    rate_limit: dict[str, JSONValue] | None = None
    seed_requested: int | None = None
    seed_applied: int | None = None
    seed_capability: SeedCapability = SeedCapability.UNSUPPORTED


class StageTelemetry(HashableModel):
    """Aggregated provider-call telemetry for one execution stage."""

    stage: str
    provider_call_count: int = 0
    provider_failure_count: int = 0
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)


class GenerationTurn(HashableModel):
    """One turn in a candidate generation execution."""

    turn_index: int
    input_messages: list[Message] = Field(default_factory=list)
    output_messages: list[Message] = Field(default_factory=list)
    artifacts: dict[str, JSONValue] = Field(default_factory=dict)
    trace: list[TraceStep] = Field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class Candidate(HashableModel):
    """A generated candidate, including optional multi-turn evidence."""

    candidate_id: str
    final_output: JSONValue
    turns: list[GenerationTurn] = Field(default_factory=list)
    stream_events: list[StreamEvent] = Field(default_factory=list)
    environment_state: dict[str, JSONValue] = Field(default_factory=dict)
    termination_reason: str | None = None
    trace: list[TraceStep] | None = None
    conversation: list[Message] | None = None
    artifacts: dict[str, JSONValue] | None = None
    token_usage: dict[str, int] | None = None
    latency_ms: float | None = None


class ParsedOutput(HashableModel):
    """Normalized output produced by a parser before scoring."""

    value: JSONValue
    format: str | None = None
    confidence: float | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class FailureCategory(StrEnum):
    """Stable failure categories surfaced by projections and reports."""

    PARSE_FAILURE = "parse_failure"
    PROVIDER_FAILURE = "provider_failure"
    JUDGE_PARSE_FAILURE = "judge_parse_failure"
    RUBRIC_AMBIGUITY = "rubric_ambiguity"
    DATA_PROBLEM = "data_problem"
    METRIC_FAILURE = "metric_failure"


class MetricResult(HashableModel):
    """Successful metric output with a numeric summary and rich metadata."""

    metric_id: str
    result_type: Literal[
        "scalar", "rubric", "preference", "ranking", "agreement", "calibration"
    ] = "scalar"
    value: float | None = None
    dimensions: dict[str, float] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    confidence: float | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ScoreError(HashableModel):
    """Structured score failure recorded by the runtime."""

    metric_id: str
    reason: str
    category: FailureCategory = FailureCategory.METRIC_FAILURE
    retryable: bool = False
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ReducedCandidate(HashableModel):
    """Candidate selected or synthesized by the reduction stage."""

    candidate_id: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    final_output: JSONValue
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class WorkflowTrace(HashableModel):
    """Trace emitted by a workflow-backed evaluation."""

    trace_id: str
    steps: list[TraceStep] = Field(default_factory=list)


class ConversationTrace(HashableModel):
    """Conversation trace captured during generation."""

    trace_id: str
    messages: list[Message] = Field(default_factory=list)
