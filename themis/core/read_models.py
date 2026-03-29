"""Projection-backed read models for the Phase 4 read side."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from themis.core.base import FrozenModel
from themis.core.workflows import EvaluationExecution


class BenchmarkScoreRow(FrozenModel):
    case_id: str
    metric_id: str
    value: float
    candidate_id: str | None = None


class BenchmarkResult(FrozenModel):
    run_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0
    score_rows: list[BenchmarkScoreRow] = Field(default_factory=list)
    metric_means: dict[str, float] = Field(default_factory=dict)


class TimelineEntry(FrozenModel):
    index: int
    event_type: str
    occurred_at: datetime
    case_id: str | None = None
    candidate_id: str | None = None
    metric_id: str | None = None


class TimelineView(FrozenModel):
    run_id: str
    entries: list[TimelineEntry] = Field(default_factory=list)


class GenerationTraceRecord(FrozenModel):
    case_id: str
    candidate_id: str
    trace_id: str
    steps: list[dict[str, object]] = Field(default_factory=list)


class ConversationTraceRecord(FrozenModel):
    case_id: str
    candidate_id: str
    trace_id: str
    messages: list[dict[str, object]] = Field(default_factory=list)


class EvaluationTraceRecord(FrozenModel):
    case_id: str
    metric_id: str
    candidate_id: str | None = None
    execution: EvaluationExecution


class TraceView(FrozenModel):
    run_id: str
    generation_traces: list[GenerationTraceRecord] = Field(default_factory=list)
    conversation_traces: list[ConversationTraceRecord] = Field(default_factory=list)
    evaluation_traces: list[EvaluationTraceRecord] = Field(default_factory=list)
