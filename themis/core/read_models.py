"""Projection-backed read models for the Phase 4 read side."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from themis.core.base import FrozenModel, JSONValue
from themis.core.workflows import EvaluationExecution


class BenchmarkScoreRow(FrozenModel):
    """One score row in the benchmark projection."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    metric_id: str
    value: float | None = None
    candidate_id: str | None = None
    outcome: Literal["correct", "incorrect", "error"] = "incorrect"
    error_category: str | None = None
    error_message: str | None = None
    details: dict[str, JSONValue] = Field(default_factory=dict)


class BenchmarkResult(FrozenModel):
    """Aggregate benchmark-style projection for a run."""

    run_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0
    score_rows: list[BenchmarkScoreRow] = Field(default_factory=list)
    metric_means: dict[str, float] = Field(default_factory=dict)
    outcome_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    error_counts: dict[str, dict[str, int]] = Field(default_factory=dict)


class TimelineEntry(FrozenModel):
    """One chronological event entry in the timeline projection."""

    index: int
    event_type: str
    occurred_at: datetime
    case_id: str | None = None
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str | None = None
    metric_id: str | None = None


class TimelineView(FrozenModel):
    """Timeline projection for a run."""

    run_id: str
    entries: list[TimelineEntry] = Field(default_factory=list)


class GenerationTraceRecord(FrozenModel):
    """One generation trace record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    trace_id: str
    steps: list[dict[str, object]] = Field(default_factory=list)


class ConversationTraceRecord(FrozenModel):
    """One conversation trace record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    trace_id: str
    messages: list[dict[str, object]] = Field(default_factory=list)


class EvaluationTraceRecord(FrozenModel):
    """One evaluation trace record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    metric_id: str
    candidate_id: str | None = None
    execution: EvaluationExecution


class TraceView(FrozenModel):
    """Trace-oriented projection for a run."""

    run_id: str
    generation_traces: list[GenerationTraceRecord] = Field(default_factory=list)
    conversation_traces: list[ConversationTraceRecord] = Field(default_factory=list)
    evaluation_traces: list[EvaluationTraceRecord] = Field(default_factory=list)
