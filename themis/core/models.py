"""Core immutable domain models for Themis v4."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from themis.core.base import HashableModel, JSONValue


class Case(HashableModel):
    case_id: str
    input: JSONValue
    expected_output: JSONValue | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Dataset(HashableModel):
    dataset_id: str
    cases: list[Case] = Field(default_factory=list)
    revision: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Message(HashableModel):
    role: str
    content: JSONValue


class TraceStep(HashableModel):
    step_name: str
    step_type: str
    input: dict[str, JSONValue] = Field(default_factory=dict)
    output: dict[str, JSONValue] = Field(default_factory=dict)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    timestamp: datetime | None = None


class GenerationResult(HashableModel):
    candidate_id: str
    final_output: JSONValue
    trace: list[TraceStep] | None = None
    conversation: list[Message] | None = None
    artifacts: dict[str, JSONValue] | None = None
    token_usage: dict[str, int] | None = None
    latency_ms: float | None = None


class ParsedOutput(HashableModel):
    value: JSONValue
    format: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class Score(HashableModel):
    metric_id: str
    value: float
    details: dict[str, JSONValue] = Field(default_factory=dict)


class ScoreError(HashableModel):
    metric_id: str
    reason: str
    retryable: bool = False
    details: dict[str, JSONValue] = Field(default_factory=dict)


class ReducedCandidate(HashableModel):
    candidate_id: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    final_output: JSONValue
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class WorkflowTrace(HashableModel):
    trace_id: str
    steps: list[TraceStep] = Field(default_factory=list)


class ConversationTrace(HashableModel):
    trace_id: str
    messages: list[Message] = Field(default_factory=list)
