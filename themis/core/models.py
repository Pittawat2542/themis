"""Core immutable domain models for Themis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from themis.core.base import HashableModel, JSONValue


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


class GenerationResult(HashableModel):
    """The candidate artifact returned by a generator call."""

    candidate_id: str
    final_output: JSONValue
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
