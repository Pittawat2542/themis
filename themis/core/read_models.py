"""Projection-backed read models for persisted inspection and reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from themis.core.base import FrozenModel, JSONValue
from themis.core.models import (
    MetricResult,
    ParsedOutput,
    ReducedCandidate,
    ScoreError,
    SessionResult,
)
from themis.core.workflows import EvaluationExecution


class BenchmarkScoreRow(FrozenModel):
    """One metric_result row in the benchmark projection."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    metric_id: str
    result_type: str | None = None
    value: float | None = None
    confidence: float | None = None
    dimensions: dict[str, float] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    candidate_id: str | None = None
    outcome: Literal["correct", "incorrect", "error"] = "incorrect"
    failure_category: str | None = None
    error_message: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


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


class StreamTraceRecord(FrozenModel):
    """One recorded streaming event in the trace projection."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str | None = None
    metric_id: str | None = None
    source_stage: str
    event: dict[str, JSONValue]


class TraceView(FrozenModel):
    """Trace-oriented projection for a run."""

    run_id: str
    generation_traces: list[GenerationTraceRecord] = Field(default_factory=list)
    conversation_traces: list[ConversationTraceRecord] = Field(default_factory=list)
    evaluation_traces: list[EvaluationTraceRecord] = Field(default_factory=list)
    stream_traces: list[StreamTraceRecord] = Field(default_factory=list)


class TelemetryBreakdown(FrozenModel):
    """Aggregated telemetry for one persisted artifact."""

    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    request_ids: list[str] = Field(default_factory=list)
    retry_count: int = 0
    estimated_cost: float = 0.0


class GenerationAuditRecord(FrozenModel):
    """Case-scoped audit record for one generation attempt."""

    candidate_id: str
    candidate_index: int | None = None
    result: SessionResult
    telemetry: TelemetryBreakdown = Field(default_factory=TelemetryBreakdown)


class MetricAuditRecord(FrozenModel):
    """Case-scoped audit record for one metric."""

    metric_id: str
    metric_result: MetricResult | None = None
    score_error: ScoreError | None = None
    evaluation_execution: EvaluationExecution | None = None
    evaluation_failure: str | None = None
    evaluation_input: dict[str, JSONValue] = Field(default_factory=dict)
    failure_records: list[str] = Field(default_factory=list)
    telemetry: TelemetryBreakdown = Field(default_factory=TelemetryBreakdown)


class CaseAuditRecord(FrozenModel):
    """Unified pipeline audit record for one case."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    generation_attempts: list[GenerationAuditRecord] = Field(default_factory=list)
    generation_failures: dict[str, str] = Field(default_factory=dict)
    selected_candidate_ids: list[str] | None = None
    selection_metadata: dict[str, object] = Field(default_factory=dict)
    selection_error: str | None = None
    reduced_candidate: ReducedCandidate | None = None
    reduction_source_candidate_ids: list[str] = Field(default_factory=list)
    reduction_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    reduction_error: str | None = None
    parse_candidate_id: str | None = None
    parse_input: dict[str, JSONValue] = Field(default_factory=dict)
    parsed_views: dict[str, ParsedOutput] = Field(default_factory=dict)
    parse_errors: dict[str, str] = Field(default_factory=dict)
    metric_records: list[MetricAuditRecord] = Field(default_factory=list)


class CaseAuditView(FrozenModel):
    """Case-audit projection for a run."""

    run_id: str
    cases: list[CaseAuditRecord] = Field(default_factory=list)


class TelemetrySummary(FrozenModel):
    """Run-level telemetry summary derived from persisted case audits."""

    run_id: str
    generation_tokens: dict[str, int] = Field(default_factory=dict)
    judge_tokens: dict[str, int] = Field(default_factory=dict)
    generation_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    request_ids: list[str] = Field(default_factory=list)
    retry_count: int = 0
    estimated_cost: float = 0.0
    provider_call_count: int = 0
    provider_failure_count: int = 0
    provider_calls_by_stage: dict[str, int] = Field(default_factory=dict)
    provider_calls_by_provider: dict[str, int] = Field(default_factory=dict)
    failure_categories: dict[str, int] = Field(default_factory=dict)


class FailureSlice(FrozenModel):
    """One grouped failure slice in a benchmark result."""

    dimension: str
    value: str
    count: int
    case_keys: list[str] = Field(default_factory=list)


class FailureSliceSummary(FrozenModel):
    """Failure slices for a persisted run."""

    run_id: str
    slices: list[FailureSlice] = Field(default_factory=list)


class ReliabilitySummary(FrozenModel):
    """Reliability metrics derived from scored benchmark rows."""

    run_id: str
    metrics: list[MetricResult] = Field(default_factory=list)


class TrendPoint(FrozenModel):
    """One metric value for one persisted run in a trend view."""

    run_id: str
    metric_id: str
    value: float
    baseline_label: str | None = None
    created_at: datetime


class TrendView(FrozenModel):
    """Metric trend over persisted run records."""

    metric_id: str
    points: list[TrendPoint] = Field(default_factory=list)


class RegressionFinding(FrozenModel):
    """One threshold comparison between a baseline and candidate metric."""

    metric_id: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_value: float
    candidate_value: float
    delta: float
    threshold: float
    regressed: bool


class RegressionSummary(FrozenModel):
    """Threshold regression findings for one candidate run."""

    candidate_run_id: str
    baseline_label: str
    findings: list[RegressionFinding] = Field(default_factory=list)
