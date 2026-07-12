"""Projection builders for Phase 4 read models."""

from __future__ import annotations

from collections import defaultdict

from themis.core.base import JSONValue
from themis.core.case_refs import CaseRef, resolve_case_key
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ProviderCallCompletedEvent,
    ProviderCallFailedEvent,
    RunEvent,
    StreamRecordedEvent,
)
from themis.core.models import (
    Candidate,
    MetricDirection,
    MetricInterpretation,
    MetricResult,
    ScoreError,
    ScoreOutcome,
)
from themis.core.read_models import (
    BenchmarkResult,
    BenchmarkScoreRow,
    CaseAuditRecord,
    CaseAuditView,
    ConversationTraceRecord,
    EvaluationTraceRecord,
    GenerationAuditRecord,
    GenerationTraceRecord,
    MetricAuditRecord,
    StreamTraceRecord,
    TelemetryBreakdown,
    TelemetrySummary,
    TimelineEntry,
    TimelineView,
    TraceView,
)
from themis.core.results import (
    CaseExecutionState,
    CaseResult,
    ExecutionState,
    ProgressSnapshot,
    RunResult,
    _case_state_has_failures,
)
from themis.core.snapshot import RunSnapshot
from themis.core.workflows import EvaluationExecution

PROJECTION_NAMES = (
    "snapshot",
    "run_result",
    "benchmark_result",
    "timeline_view",
    "trace_view",
    "case_audit_view",
    "telemetry_summary",
)
STORE_PROJECTION_NAMES = PROJECTION_NAMES + ("execution_state",)


def build_run_result(snapshot: RunSnapshot, events: list[RunEvent]) -> RunResult:
    return build_run_result_from_state(
        snapshot, ExecutionState.from_events(snapshot.run_id, events)
    )


def build_run_result_from_state(
    snapshot: RunSnapshot, state: ExecutionState
) -> RunResult:
    case_results: list[CaseResult] = []
    failed_cases = 0
    completed_cases = 0
    case_identities = _snapshot_case_identities(snapshot)

    for dataset in snapshot.datasets:
        for case in dataset.cases:
            case_ref = CaseRef(dataset_id=dataset.dataset_id, case_id=case.case_id)
            case_state = _case_state_for_snapshot_case(state, case_ref, case_identities)
            if case_state is None:
                case_results.append(
                    CaseResult(
                        case_id=case.case_id,
                        dataset_id=dataset.dataset_id,
                        case_key=case_ref.case_key,
                    )
                )
                continue

            has_failure = _case_state_has_failures(case_state)
            failed_cases += int(has_failure)
            completed_cases += int(not has_failure)
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    dataset_id=dataset.dataset_id,
                    case_key=case_ref.case_key,
                    generated_candidates=[
                        case_state.generated_candidates_by_index[index]
                        for index in sorted(case_state.generated_candidates_by_index)
                    ]
                    or list(case_state.generated_candidates.values()),
                    generated_candidate_blob_refs=dict(
                        case_state.generated_candidate_blob_refs
                    ),
                    generation_failures=dict(case_state.generation_failures),
                    reduced_candidate=case_state.reduced_candidate,
                    reduction_error=case_state.reduction_error,
                    parsed_views=case_state.parsed_views,
                    parse_errors=case_state.parse_errors,
                    evaluation_executions=list(
                        case_state.evaluation_executions.values()
                    ),
                    evaluation_execution_blob_refs=dict(
                        case_state.evaluation_execution_blob_refs
                    ),
                    evaluation_failures=dict(case_state.evaluation_failures),
                    metric_results=list(case_state.metric_results.values())
                    + list(case_state.score_failures.values()),
                )
            )

    total_cases = sum(len(dataset.cases) for dataset in snapshot.datasets)
    return RunResult(
        run_id=snapshot.run_id,
        status=state.status,
        completed_through_stage=state.completed_through_stage,
        progress=ProgressSnapshot(
            total_cases=total_cases,
            completed_cases=completed_cases,
            failed_cases=failed_cases,
        ),
        cases=case_results,
    )


def build_benchmark_result(
    snapshot: RunSnapshot, events: list[RunEvent]
) -> BenchmarkResult:
    benchmark_result = build_benchmark_result_from_run_result(
        build_run_result(snapshot, events),
        metric_ids=[
            metric_ref.component_id for metric_ref in snapshot.component_refs.metrics
        ],
        metric_interpretations={
            metric_ref.component_id: metric_ref.interpretation
            for metric_ref in snapshot.component_refs.metrics
        },
    )
    return benchmark_result.model_copy(
        update={"dataset_ids": [dataset.dataset_id for dataset in snapshot.datasets]}
    )


def build_benchmark_result_from_run_result(
    run_result: RunResult,
    *,
    metric_ids: list[str] | None = None,
    metric_interpretations: dict[str, MetricInterpretation] | None = None,
) -> BenchmarkResult:
    score_rows: list[BenchmarkScoreRow] = []
    metric_scores: dict[str, list[float]] = defaultdict(list)
    outcome_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    error_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    explicit_metric_ids = set(metric_ids or [])
    known_metric_ids = set(explicit_metric_ids)

    for case in run_result.cases:
        candidate_id = (
            case.reduced_candidate.candidate_id
            if case.reduced_candidate is not None
            else None
        )
        metric_results = {
            metric_result.metric_id: metric_result
            for metric_result in case.metric_results
            if isinstance(metric_result, MetricResult)
        }
        score_errors = {
            metric_result.metric_id: metric_result
            for metric_result in case.metric_results
            if isinstance(metric_result, ScoreError)
        }
        execution_failures = _execution_failures_by_metric(case)
        case_metric_ids = explicit_metric_ids or set(metric_results) | set(
            score_errors
        ) | set(case.evaluation_failures) | set(execution_failures)
        known_metric_ids.update(case_metric_ids)

        for metric_id in sorted(case_metric_ids):
            row = _benchmark_row_for_metric(
                case=case,
                metric_id=metric_id,
                candidate_id=candidate_id,
                metric_results=metric_results,
                score_errors=score_errors,
                execution_failures=execution_failures,
                interpretation=(metric_interpretations or {}).get(
                    metric_id, MetricInterpretation()
                ),
            )
            if row is None:
                continue
            score_rows.append(row)
            outcome_counts[row.metric_id][row.outcome] += 1
            if row.outcome != "error" and row.value is not None:
                metric_scores[row.metric_id].append(row.value)
            if row.failure_category is not None:
                error_counts[row.metric_id][row.failure_category] += 1

    return BenchmarkResult(
        run_id=run_result.run_id,
        dataset_ids=[],
        metric_ids=sorted(known_metric_ids or metric_scores),
        total_cases=run_result.progress.total_cases,
        completed_cases=run_result.progress.completed_cases,
        failed_cases=run_result.progress.failed_cases,
        score_rows=score_rows,
        metric_means={
            metric_id: sum(values) / len(values)
            for metric_id, values in metric_scores.items()
            if values
        },
        outcome_counts={
            metric_id: dict(counts)
            for metric_id, counts in sorted(outcome_counts.items())
        },
        error_counts={
            metric_id: dict(counts)
            for metric_id, counts in sorted(error_counts.items())
        },
    )


def build_timeline_view(snapshot: RunSnapshot, events: list[RunEvent]) -> TimelineView:
    case_identities = _snapshot_case_identities(snapshot)
    return TimelineView(
        run_id=snapshot.run_id,
        entries=[
            TimelineEntry(
                index=index,
                event_type=event.event_type,
                occurred_at=event.occurred_at,
                case_id=getattr(event, "case_id", None),
                dataset_id=_event_dataset_id(event, case_identities),
                case_key=_event_case_key(event, case_identities),
                candidate_id=getattr(event, "candidate_id", None),
                metric_id=getattr(event, "metric_id", None),
            )
            for index, event in enumerate(events)
        ],
    )


def build_trace_view(snapshot: RunSnapshot, events: list[RunEvent]) -> TraceView:
    view = TraceView(run_id=snapshot.run_id)
    for event in events:
        view = _apply_event_to_trace_view(snapshot, view, event)
    return view


def _generation_telemetry(result: Candidate) -> TelemetryBreakdown:
    artifacts = result.artifacts or {}
    request_ids = [
        str(value)
        for key in ("provider_request_id", "request_id")
        if (value := artifacts.get(key)) is not None
    ]
    retry_history = artifacts.get("retry_history", [])
    retry_count = len(retry_history) if isinstance(retry_history, list) else 0
    estimated_cost = artifacts.get("cost_estimate", artifacts.get("cost_usd", 0.0))
    return TelemetryBreakdown(
        token_usage=dict(result.token_usage or {}),
        latency_ms=float(result.latency_ms or 0.0),
        request_ids=request_ids,
        retry_count=retry_count,
        estimated_cost=float(estimated_cost)
        if isinstance(estimated_cost, (int, float))
        else 0.0,
    )


def _evaluation_telemetry(execution: EvaluationExecution | None) -> TelemetryBreakdown:
    if execution is None:
        return TelemetryBreakdown()
    token_usage: dict[str, int] = {}
    latency_ms = 0.0
    request_ids: list[str] = []
    retry_count = 0
    for response in execution.judge_responses:
        token_usage = _merge_token_usage(token_usage, dict(response.token_usage))
        latency_ms += float(response.latency_ms or 0.0)
        if response.provider_request_id is not None:
            request_ids.append(response.provider_request_id)
        retry_count += len(response.retry_history)
    return TelemetryBreakdown(
        token_usage=token_usage,
        latency_ms=latency_ms,
        request_ids=request_ids,
        retry_count=retry_count,
    )


def _merge_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def build_case_audit_view(
    snapshot: RunSnapshot, events: list[RunEvent]
) -> CaseAuditView:
    state = ExecutionState.from_events(snapshot.run_id, events)
    return build_case_audit_view_from_state(snapshot, state)


def build_case_audit_view_from_state(
    snapshot: RunSnapshot, state: ExecutionState
) -> CaseAuditView:
    case_identities = _snapshot_case_identities(snapshot)
    audits: list[CaseAuditRecord] = []
    for dataset in snapshot.datasets:
        for case in dataset.cases:
            case_ref = CaseRef(dataset_id=dataset.dataset_id, case_id=case.case_id)
            case_state = _case_state_for_snapshot_case(state, case_ref, case_identities)
            if case_state is None:
                audits.append(
                    CaseAuditRecord(
                        case_id=case.case_id,
                        dataset_id=dataset.dataset_id,
                        case_key=case_ref.case_key,
                    )
                )
                continue
            generation_attempts = [
                GenerationAuditRecord(
                    candidate_id=result.candidate_id,
                    candidate_index=index,
                    result=result,
                    telemetry=_generation_telemetry(result),
                )
                for index, result in sorted(
                    case_state.generated_candidates_by_index.items()
                )
            ] or [
                GenerationAuditRecord(
                    candidate_id=result.candidate_id,
                    result=result,
                    telemetry=_generation_telemetry(result),
                )
                for result in case_state.generated_candidates.values()
            ]
            metric_ids = sorted(
                set(case_state.metric_results)
                | set(case_state.score_failures)
                | set(case_state.evaluation_executions)
                | set(case_state.evaluation_failures)
            )
            metric_records = [
                MetricAuditRecord(
                    metric_id=metric_id,
                    metric_result=case_state.metric_results.get(metric_id),
                    score_error=case_state.score_failures.get(metric_id),
                    evaluation_execution=case_state.evaluation_executions.get(
                        metric_id
                    ),
                    evaluation_failure=case_state.evaluation_failures.get(metric_id),
                    evaluation_input=(
                        {"candidate_id": case_state.reduced_candidate.candidate_id}
                        if case_state.reduced_candidate is not None
                        else {}
                    ),
                    failure_records=(
                        [case_state.evaluation_failures[metric_id]]
                        if metric_id in case_state.evaluation_failures
                        else []
                    ),
                    telemetry=_evaluation_telemetry(
                        case_state.evaluation_executions.get(metric_id)
                    ),
                )
                for metric_id in metric_ids
            ]
            audits.append(
                CaseAuditRecord(
                    case_id=case.case_id,
                    dataset_id=dataset.dataset_id,
                    case_key=case_ref.case_key,
                    generation_attempts=generation_attempts,
                    generation_failures=dict(case_state.generation_failures),
                    selected_candidate_ids=case_state.selected_candidate_ids,
                    selection_metadata=dict(case_state.selection_metadata),
                    selection_error=case_state.selection_error,
                    reduced_candidate=case_state.reduced_candidate,
                    reduction_source_candidate_ids=(
                        []
                        if case_state.reduced_candidate is None
                        else list(case_state.reduced_candidate.source_candidate_ids)
                    ),
                    reduction_metadata=(
                        {}
                        if case_state.reduced_candidate is None
                        else dict(case_state.reduced_candidate.metadata)
                    ),
                    reduction_error=case_state.reduction_error,
                    parse_candidate_id=(
                        None
                        if case_state.reduced_candidate is None
                        else case_state.reduced_candidate.candidate_id
                    ),
                    parse_input=(
                        {}
                        if case_state.reduced_candidate is None
                        else dict(case_state.reduced_candidate.final_output)
                        if isinstance(case_state.reduced_candidate.final_output, dict)
                        else {"value": case_state.reduced_candidate.final_output}
                    ),
                    parsed_views=case_state.parsed_views,
                    parse_errors=case_state.parse_errors,
                    metric_records=metric_records,
                )
            )
    return CaseAuditView(run_id=snapshot.run_id, cases=audits)


def build_telemetry_summary(
    snapshot: RunSnapshot, events: list[RunEvent]
) -> TelemetrySummary:
    summary = build_telemetry_summary_from_case_audit(
        snapshot.run_id, build_case_audit_view(snapshot, events)
    )
    return _with_provider_call_telemetry(summary, events)


def build_telemetry_summary_from_case_audit(
    run_id: str, case_audit_view: CaseAuditView
) -> TelemetrySummary:
    generation_tokens: dict[str, int] = {}
    judge_tokens: dict[str, int] = {}
    generation_latency_ms = 0.0
    judge_latency_ms = 0.0
    request_ids: list[str] = []
    retry_count = 0
    estimated_cost = 0.0
    for case in case_audit_view.cases:
        for attempt in case.generation_attempts:
            generation_tokens = _merge_token_usage(
                generation_tokens, attempt.telemetry.token_usage
            )
            generation_latency_ms += attempt.telemetry.latency_ms
            request_ids.extend(attempt.telemetry.request_ids)
            retry_count += attempt.telemetry.retry_count
            estimated_cost += attempt.telemetry.estimated_cost
        for metric_record in case.metric_records:
            judge_tokens = _merge_token_usage(
                judge_tokens, metric_record.telemetry.token_usage
            )
            judge_latency_ms += metric_record.telemetry.latency_ms
            request_ids.extend(metric_record.telemetry.request_ids)
            retry_count += metric_record.telemetry.retry_count
            estimated_cost += metric_record.telemetry.estimated_cost
    return TelemetrySummary(
        run_id=run_id,
        generation_tokens=generation_tokens,
        judge_tokens=judge_tokens,
        generation_latency_ms=generation_latency_ms,
        judge_latency_ms=judge_latency_ms,
        request_ids=sorted(dict.fromkeys(request_ids)),
        retry_count=retry_count,
        estimated_cost=estimated_cost,
    )


def _with_provider_call_telemetry(
    summary: TelemetrySummary, events: list[RunEvent]
) -> TelemetrySummary:
    provider_call_count = 0
    provider_failure_count = 0
    provider_calls_by_stage: dict[str, int] = {}
    provider_calls_by_provider: dict[str, int] = {}
    failure_categories: dict[str, int] = {}
    for event in events:
        if isinstance(event, ProviderCallCompletedEvent):
            provider_call_count += 1
            provider_calls_by_stage[event.stage] = (
                provider_calls_by_stage.get(event.stage, 0) + 1
            )
            provider_calls_by_provider[event.provider_id] = (
                provider_calls_by_provider.get(event.provider_id, 0) + 1
            )
        elif isinstance(event, ProviderCallFailedEvent):
            provider_failure_count += 1
            failure_categories[event.failure_category] = (
                failure_categories.get(event.failure_category, 0) + 1
            )
    return summary.model_copy(
        update={
            "provider_call_count": provider_call_count,
            "provider_failure_count": provider_failure_count,
            "provider_calls_by_stage": provider_calls_by_stage,
            "provider_calls_by_provider": provider_calls_by_provider,
            "failure_categories": failure_categories,
        }
    )


def build_projection_payloads(
    snapshot: RunSnapshot, events: list[RunEvent]
) -> dict[str, JSONValue]:
    store_payloads = build_store_projection_payloads(snapshot, events)
    return {
        projection_name: store_payloads[projection_name]
        for projection_name in PROJECTION_NAMES
    }


def build_store_projection_payloads(
    snapshot: RunSnapshot, events: list[RunEvent]
) -> dict[str, JSONValue]:
    state = ExecutionState.from_events(snapshot.run_id, events)
    run_result = build_run_result_from_state(snapshot, state)
    benchmark_result = build_benchmark_result_from_run_result(
        run_result,
        metric_interpretations={
            ref.component_id: ref.interpretation
            for ref in snapshot.component_refs.metrics
        },
    ).model_copy(
        update={
            "dataset_ids": [dataset.dataset_id for dataset in snapshot.datasets],
            "metric_ids": [
                metric_ref.component_id
                for metric_ref in snapshot.component_refs.metrics
            ],
        }
    )
    snapshot_payload = snapshot.model_dump(mode="json")
    return {
        "snapshot": snapshot_payload,
        "execution_state": state.model_dump(mode="json"),
        "run_result": run_result.model_dump(mode="json"),
        "benchmark_result": benchmark_result.model_dump(mode="json"),
        "timeline_view": build_timeline_view(snapshot, events).model_dump(mode="json"),
        "trace_view": build_trace_view(snapshot, events).model_dump(mode="json"),
        "case_audit_view": build_case_audit_view(snapshot, events).model_dump(
            mode="json"
        ),
        "telemetry_summary": build_telemetry_summary(snapshot, events).model_dump(
            mode="json"
        ),
    }


def build_initial_store_projection_payloads(
    snapshot: RunSnapshot,
) -> dict[str, JSONValue]:
    return build_store_projection_payloads(snapshot, [])


def apply_event_to_store_projection_payloads(
    snapshot: RunSnapshot,
    projections: dict[str, JSONValue],
    event: RunEvent,
) -> dict[str, JSONValue]:
    state = _current_execution_state(snapshot, projections).apply_event(event)
    run_result = build_run_result_from_state(snapshot, state)
    benchmark_result = build_benchmark_result_from_run_result(
        run_result,
        metric_interpretations={
            ref.component_id: ref.interpretation
            for ref in snapshot.component_refs.metrics
        },
    ).model_copy(
        update={
            "dataset_ids": [dataset.dataset_id for dataset in snapshot.datasets],
            "metric_ids": [
                metric_ref.component_id
                for metric_ref in snapshot.component_refs.metrics
            ],
        }
    )
    snapshot_payload = _current_snapshot_payload(snapshot, projections.get("snapshot"))
    timeline_view = _apply_event_to_timeline_view(
        snapshot, projections.get("timeline_view"), event
    )
    trace_view = _apply_event_to_trace_view(
        snapshot,
        _current_trace_view(snapshot, projections.get("trace_view")),
        event,
    )
    case_audit_view = build_case_audit_view_from_state(snapshot, state)
    telemetry_summary = build_telemetry_summary_from_case_audit(
        snapshot.run_id,
        case_audit_view,
    )
    telemetry_summary = _merge_provider_summary_fields(
        telemetry_summary,
        _current_telemetry_summary(snapshot, projections.get("telemetry_summary")),
    )
    telemetry_summary = _apply_provider_event_to_summary(telemetry_summary, event)
    return {
        "snapshot": snapshot_payload,
        "execution_state": state.model_dump(mode="json"),
        "run_result": run_result.model_dump(mode="json"),
        "benchmark_result": benchmark_result.model_dump(mode="json"),
        "timeline_view": timeline_view.model_dump(mode="json"),
        "trace_view": trace_view.model_dump(mode="json"),
        "case_audit_view": case_audit_view.model_dump(mode="json"),
        "telemetry_summary": telemetry_summary.model_dump(mode="json"),
    }


def _current_telemetry_summary(
    snapshot: RunSnapshot, payload: JSONValue | None
) -> TelemetrySummary:
    if isinstance(payload, dict):
        return TelemetrySummary.model_validate(payload)
    return TelemetrySummary(run_id=snapshot.run_id)


def _merge_provider_summary_fields(
    summary: TelemetrySummary, existing: TelemetrySummary
) -> TelemetrySummary:
    return summary.model_copy(
        update={
            "provider_call_count": existing.provider_call_count,
            "provider_failure_count": existing.provider_failure_count,
            "provider_calls_by_stage": dict(existing.provider_calls_by_stage),
            "provider_calls_by_provider": dict(existing.provider_calls_by_provider),
            "failure_categories": dict(existing.failure_categories),
        }
    )


def _apply_provider_event_to_summary(
    summary: TelemetrySummary, event: RunEvent
) -> TelemetrySummary:
    if isinstance(event, ProviderCallCompletedEvent):
        by_stage = dict(summary.provider_calls_by_stage)
        by_provider = dict(summary.provider_calls_by_provider)
        by_stage[event.stage] = by_stage.get(event.stage, 0) + 1
        by_provider[event.provider_id] = by_provider.get(event.provider_id, 0) + 1
        return summary.model_copy(
            update={
                "provider_call_count": summary.provider_call_count + 1,
                "provider_calls_by_stage": by_stage,
                "provider_calls_by_provider": by_provider,
            }
        )
    if isinstance(event, ProviderCallFailedEvent):
        failure_categories = dict(summary.failure_categories)
        failure_categories[event.failure_category] = (
            failure_categories.get(event.failure_category, 0) + 1
        )
        return summary.model_copy(
            update={
                "provider_failure_count": summary.provider_failure_count + 1,
                "failure_categories": failure_categories,
            }
        )
    return summary


def _benchmark_row_for_metric(
    *,
    case: CaseResult,
    metric_id: str,
    candidate_id: str | None,
    metric_results: dict[str, MetricResult],
    score_errors: dict[str, ScoreError],
    execution_failures: dict[str, str],
    interpretation: MetricInterpretation,
) -> BenchmarkScoreRow | None:
    if metric_id in case.evaluation_failures:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            case_key=case.case_key,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome=ScoreOutcome.ERROR,
            failure_category="evaluation_failure",
            error_message=case.evaluation_failures[metric_id],
        )
    if metric_id in execution_failures:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            case_key=case.case_key,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome=ScoreOutcome.ERROR,
            failure_category="evaluation_partial_failure",
            error_message=execution_failures[metric_id],
        )
    if metric_id in score_errors:
        score_error = score_errors[metric_id]
        return BenchmarkScoreRow(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            case_key=case.case_key,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome=ScoreOutcome.ERROR,
            failure_category=str(score_error.category),
            error_message=score_error.reason,
            metadata=dict(score_error.metadata),
        )
    if metric_id in metric_results:
        metric_result = metric_results[metric_id]
        validation_error = _metric_value_error(metric_result, interpretation)
        if validation_error is not None:
            return BenchmarkScoreRow(
                case_id=case.case_id,
                dataset_id=case.dataset_id,
                case_key=case.case_key,
                metric_id=metric_id,
                candidate_id=candidate_id,
                outcome=ScoreOutcome.ERROR,
                failure_category="metric_value_invalid",
                error_message=validation_error,
            )
        return BenchmarkScoreRow(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            case_key=case.case_key,
            metric_id=metric_id,
            result_type=metric_result.result_type,
            value=float(metric_result.value)
            if metric_result.value is not None
            else None,
            confidence=metric_result.confidence,
            dimensions=dict(metric_result.dimensions),
            labels=dict(metric_result.labels),
            candidate_id=candidate_id,
            outcome=_score_outcome(metric_result, interpretation),
            metadata=dict(metric_result.metadata),
        )
    parse_errors_row = _parse_errors_row(case, metric_id, candidate_id)
    if parse_errors_row is not None:
        return parse_errors_row
    return None


def _parse_errors_row(
    case: CaseResult,
    metric_id: str,
    candidate_id: str | None,
) -> BenchmarkScoreRow | None:
    if case.parse_errors:
        parser_view, message = next(iter(case.parse_errors.items()))
        return BenchmarkScoreRow(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            case_key=case.case_key,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome=ScoreOutcome.ERROR,
            failure_category="parse_failure",
            error_message=message,
            metadata={"parser_view": parser_view},
        )
    if not case.parsed_views:
        return None
    invalid_views = [
        parser_view
        for parser_view, parsed in case.parsed_views.items()
        if parsed.value is None or parsed.metadata.get("invalid") is True
    ]
    if invalid_views:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            case_key=case.case_key,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome=ScoreOutcome.ERROR,
            failure_category="parse_null",
            error_message="Parser returned null or invalid output",
            metadata={"parser_view": invalid_views[0]},
        )
    return None


def _execution_failures_by_metric(case: CaseResult) -> dict[str, str]:
    failures: dict[str, str] = {}
    for execution in case.evaluation_executions:
        if execution.status != "partial_failure" and not execution.failures:
            continue
        metric_ids = {
            metric_result.metric_id for metric_result in execution.metric_results
        }
        if not metric_ids:
            continue
        error_message = (
            "; ".join(failure.error_message for failure in execution.failures)
            or "Evaluation completed with workflow failures"
        )
        for metric_id in metric_ids:
            failures[metric_id] = error_message
    return failures


def _metric_value_error(
    metric_result: MetricResult, interpretation: MetricInterpretation
) -> str | None:
    if metric_result.value is None:
        return "Metric returned no numeric value"
    if interpretation.valid_range is None:
        return None
    lower, upper = interpretation.valid_range
    if lower <= metric_result.value <= upper:
        return None
    return f"Metric value {metric_result.value} is outside [{lower}, {upper}]"


def _score_outcome(
    metric_result: MetricResult, interpretation: MetricInterpretation
) -> ScoreOutcome:
    threshold = interpretation.correctness_threshold
    if threshold is None:
        return ScoreOutcome.SCORED
    assert metric_result.value is not None
    if interpretation.direction is MetricDirection.HIGHER_IS_BETTER:
        correct = metric_result.value >= threshold
    else:
        correct = metric_result.value <= threshold
    return ScoreOutcome.CORRECT if correct else ScoreOutcome.INCORRECT


def _current_execution_state(
    snapshot: RunSnapshot, projections: dict[str, JSONValue]
) -> ExecutionState:
    payload = projections.get("execution_state")
    if isinstance(payload, dict):
        return ExecutionState.model_validate(payload)
    return ExecutionState(run_id=snapshot.run_id)


def _current_snapshot_payload(
    snapshot: RunSnapshot, payload: JSONValue | None
) -> JSONValue:
    if isinstance(payload, dict):
        return payload
    return snapshot.model_dump(mode="json")


def _current_trace_view(snapshot: RunSnapshot, payload: JSONValue | None) -> TraceView:
    if isinstance(payload, dict):
        return TraceView.model_validate(payload)
    return TraceView(run_id=snapshot.run_id)


def _apply_event_to_timeline_view(
    snapshot: RunSnapshot,
    payload: JSONValue | None,
    event: RunEvent,
) -> TimelineView:
    case_identities = _snapshot_case_identities(snapshot)
    view = (
        TimelineView.model_validate(payload)
        if isinstance(payload, dict)
        else TimelineView(run_id=snapshot.run_id)
    )
    entries = list(view.entries)
    entries.append(
        TimelineEntry(
            index=len(entries),
            event_type=event.event_type,
            occurred_at=event.occurred_at,
            case_id=getattr(event, "case_id", None),
            dataset_id=_event_dataset_id(event, case_identities),
            case_key=_event_case_key(event, case_identities),
            candidate_id=getattr(event, "candidate_id", None),
            metric_id=getattr(event, "metric_id", None),
        )
    )
    return view.model_copy(update={"entries": entries})


def _apply_event_to_trace_view(
    snapshot: RunSnapshot, view: TraceView, event: RunEvent
) -> TraceView:
    case_identities = _snapshot_case_identities(snapshot)
    dataset_id = _event_dataset_id(event, case_identities)
    case_key = _event_case_key(event, case_identities)
    generation_traces = list(view.generation_traces)
    conversation_traces = list(view.conversation_traces)
    evaluation_traces = list(view.evaluation_traces)
    stream_traces = list(view.stream_traces)

    if isinstance(event, GenerationCompletedEvent) and event.result is not None:
        result = Candidate.model_validate(event.result)
        if result.trace:
            generation_traces.append(
                GenerationTraceRecord(
                    case_id=event.case_id,
                    dataset_id=dataset_id,
                    case_key=case_key,
                    candidate_id=event.candidate_id,
                    trace_id=f"{event.candidate_id}:generation",
                    steps=[step.model_dump(mode="json") for step in result.trace],
                )
            )
        if result.conversation:
            conversation_traces.append(
                ConversationTraceRecord(
                    case_id=event.case_id,
                    dataset_id=dataset_id,
                    case_key=case_key,
                    candidate_id=event.candidate_id,
                    trace_id=f"{event.candidate_id}:conversation",
                    messages=[
                        message.model_dump(mode="json")
                        for message in result.conversation
                    ],
                )
            )
        for turn in result.turns:
            if turn.trace:
                generation_traces.append(
                    GenerationTraceRecord(
                        case_id=event.case_id,
                        dataset_id=dataset_id,
                        case_key=case_key,
                        candidate_id=event.candidate_id,
                        trace_id=f"{event.candidate_id}:turn:{turn.turn_index}",
                        steps=[step.model_dump(mode="json") for step in turn.trace],
                    )
                )
            messages = [*turn.input_messages, *turn.output_messages]
            if messages:
                conversation_traces.append(
                    ConversationTraceRecord(
                        case_id=event.case_id,
                        dataset_id=dataset_id,
                        case_key=case_key,
                        candidate_id=event.candidate_id,
                        trace_id=f"{event.candidate_id}:turn:{turn.turn_index}:conversation",
                        messages=[
                            message.model_dump(mode="json") for message in messages
                        ],
                    )
                )
    elif isinstance(event, StreamRecordedEvent):
        stream_traces.append(
            StreamTraceRecord(
                case_id=event.case_id,
                dataset_id=dataset_id,
                case_key=case_key,
                candidate_id=event.candidate_id,
                metric_id=event.metric_id,
                source_stage=event.source_stage,
                event=event.stream_event,
            )
        )
    elif isinstance(event, EvaluationCompletedEvent) and event.execution is not None:
        evaluation_traces.append(
            EvaluationTraceRecord(
                case_id=event.case_id,
                dataset_id=dataset_id,
                case_key=case_key,
                candidate_id=event.candidate_id,
                metric_id=event.metric_id,
                execution=EvaluationExecution.model_validate(event.execution),
            )
        )

    return view.model_copy(
        update={
            "generation_traces": generation_traces,
            "conversation_traces": conversation_traces,
            "evaluation_traces": evaluation_traces,
            "stream_traces": stream_traces,
        }
    )


def _case_state_for_snapshot_case(
    state: ExecutionState,
    case_ref: CaseRef,
    identities: dict[str, CaseRef],
) -> CaseExecutionState | None:
    case_state = state.case_states.get(case_ref.case_key)
    if case_state is not None:
        return case_state
    unique_case_ref = identities.get(case_ref.case_id)
    if unique_case_ref is None or unique_case_ref.case_key != case_ref.case_key:
        return None
    return state.case_states.get(case_ref.case_id)


def _snapshot_case_identities(snapshot: RunSnapshot) -> dict[str, CaseRef]:
    identities: dict[str, CaseRef] = {}
    duplicate_case_ids: set[str] = set()
    for dataset in snapshot.datasets:
        for case in dataset.cases:
            if case.case_id in identities:
                duplicate_case_ids.add(case.case_id)
                continue
            identities[case.case_id] = CaseRef(
                dataset_id=dataset.dataset_id, case_id=case.case_id
            )
    for case_id in duplicate_case_ids:
        identities.pop(case_id, None)
    return identities


def _event_dataset_id(event: RunEvent, identities: dict[str, CaseRef]) -> str | None:
    dataset_id = getattr(event, "dataset_id", None)
    if isinstance(dataset_id, str):
        return dataset_id
    case_id = getattr(event, "case_id", None)
    if isinstance(case_id, str) and case_id in identities:
        return identities[case_id].dataset_id
    return None


def _event_case_key(event: RunEvent, identities: dict[str, CaseRef]) -> str | None:
    case_id = getattr(event, "case_id", None)
    if not isinstance(case_id, str):
        return None
    return resolve_case_key(
        case_id=case_id,
        dataset_id=_event_dataset_id(event, identities),
        case_key=getattr(event, "case_key", None),
    )
