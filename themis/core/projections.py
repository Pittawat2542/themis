"""Projection builders for Phase 4 read models."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from themis.core.base import JSONValue
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    RunEvent,
)
from themis.core.models import GenerationResult, Score, ScoreError
from themis.core.read_models import (
    BenchmarkResult,
    BenchmarkScoreRow,
    ConversationTraceRecord,
    EvaluationTraceRecord,
    GenerationTraceRecord,
    TimelineEntry,
    TimelineView,
    TraceView,
)
from themis.core.results import CaseResult, ExecutionState, ProgressSnapshot, RunResult
from themis.core.snapshot import RunSnapshot
from themis.core.workflows import EvaluationExecution

PROJECTION_NAMES = (
    "snapshot",
    "run_result",
    "benchmark_result",
    "timeline_view",
    "trace_view",
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

    for dataset in snapshot.datasets:
        for case in dataset.cases:
            case_state = state.case_states.get(case.case_id)
            if case_state is None:
                case_results.append(CaseResult(case_id=case.case_id))
                continue

            has_failure = any(
                (
                    case_state.generation_failures,
                    case_state.reduction_error is not None,
                    case_state.parse_error is not None,
                    case_state.evaluation_failures,
                    any(
                        execution.status == "partial_failure"
                        or bool(execution.failures)
                        for execution in case_state.evaluation_executions.values()
                    ),
                    case_state.score_failures,
                )
            )
            failed_cases += int(has_failure)
            completed_cases += int(not has_failure)
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
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
                    parsed_output=case_state.parsed_output,
                    parse_error=case_state.parse_error,
                    evaluation_executions=list(
                        case_state.evaluation_executions.values()
                    ),
                    evaluation_execution_blob_refs=dict(
                        case_state.evaluation_execution_blob_refs
                    ),
                    evaluation_failures=dict(case_state.evaluation_failures),
                    scores=list(case_state.successful_scores.values())
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
    )
    return benchmark_result.model_copy(
        update={"dataset_ids": [dataset.dataset_id for dataset in snapshot.datasets]}
    )


def build_benchmark_result_from_run_result(
    run_result: RunResult,
    *,
    metric_ids: list[str] | None = None,
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
        score_results = {
            score.metric_id: score for score in case.scores if isinstance(score, Score)
        }
        score_errors = {
            score.metric_id: score
            for score in case.scores
            if isinstance(score, ScoreError)
        }
        execution_failures = _execution_failures_by_metric(case)
        case_metric_ids = (
            explicit_metric_ids
            or set(score_results)
            | set(score_errors)
            | set(case.evaluation_failures)
            | set(execution_failures)
        )
        known_metric_ids.update(case_metric_ids)

        for metric_id in sorted(case_metric_ids):
            row = _benchmark_row_for_metric(
                case=case,
                metric_id=metric_id,
                candidate_id=candidate_id,
                score_results=score_results,
                score_errors=score_errors,
                execution_failures=execution_failures,
            )
            if row is None:
                continue
            score_rows.append(row)
            outcome_counts[row.metric_id][row.outcome] += 1
            if row.outcome != "error" and row.value is not None:
                metric_scores[row.metric_id].append(row.value)
            if row.error_category is not None:
                error_counts[row.metric_id][row.error_category] += 1

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
    return TimelineView(
        run_id=snapshot.run_id,
        entries=[
            TimelineEntry(
                index=index,
                event_type=event.event_type,
                occurred_at=event.occurred_at,
                case_id=getattr(event, "case_id", None),
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
    benchmark_result = build_benchmark_result_from_run_result(run_result).model_copy(
        update={
            "dataset_ids": [dataset.dataset_id for dataset in snapshot.datasets],
            "metric_ids": [
                metric_ref.component_id for metric_ref in snapshot.component_refs.metrics
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
    benchmark_result = build_benchmark_result_from_run_result(run_result).model_copy(
        update={
            "dataset_ids": [dataset.dataset_id for dataset in snapshot.datasets],
            "metric_ids": [
                metric_ref.component_id for metric_ref in snapshot.component_refs.metrics
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
    return {
        "snapshot": snapshot_payload,
        "execution_state": state.model_dump(mode="json"),
        "run_result": run_result.model_dump(mode="json"),
        "benchmark_result": benchmark_result.model_dump(mode="json"),
        "timeline_view": timeline_view.model_dump(mode="json"),
        "trace_view": trace_view.model_dump(mode="json"),
    }


def _benchmark_row_for_metric(
    *,
    case: CaseResult,
    metric_id: str,
    candidate_id: str | None,
    score_results: dict[str, Score],
    score_errors: dict[str, ScoreError],
    execution_failures: dict[str, str],
) -> BenchmarkScoreRow | None:
    parse_error_row = _parse_error_row(case, metric_id, candidate_id)
    if parse_error_row is not None:
        return parse_error_row
    if metric_id in case.evaluation_failures:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome="error",
            error_category="evaluation_failure",
            error_message=case.evaluation_failures[metric_id],
        )
    if metric_id in execution_failures:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome="error",
            error_category="evaluation_partial_failure",
            error_message=execution_failures[metric_id],
        )
    if metric_id in score_errors:
        score_error = score_errors[metric_id]
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome="error",
            error_category="score_failure",
            error_message=score_error.reason,
            details=dict(score_error.details),
        )
    if metric_id in score_results:
        score = score_results[metric_id]
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            value=float(score.value),
            candidate_id=candidate_id,
            outcome=_score_outcome(score),
            details=dict(score.details),
        )
    return None


def _parse_error_row(
    case: CaseResult,
    metric_id: str,
    candidate_id: str | None,
) -> BenchmarkScoreRow | None:
    if case.parse_error is not None:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome="error",
            error_category="parse_failure",
            error_message=case.parse_error,
        )
    if case.parsed_output is None:
        return None
    if case.parsed_output.value is None:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome="error",
            error_category="parse_null",
            error_message="Parser returned null output",
        )
    if case.parsed_output.metadata.get("invalid") is True:
        return BenchmarkScoreRow(
            case_id=case.case_id,
            metric_id=metric_id,
            candidate_id=candidate_id,
            outcome="error",
            error_category="parse_invalid",
            error_message="Parser marked output as invalid",
        )
    return None


def _execution_failures_by_metric(case: CaseResult) -> dict[str, str]:
    failures: dict[str, str] = {}
    for execution in case.evaluation_executions:
        if execution.status != "partial_failure" and not execution.failures:
            continue
        metric_ids = {score.metric_id for score in execution.scores}
        if not metric_ids:
            continue
        error_message = "; ".join(
            failure.error_message for failure in execution.failures
        ) or "Evaluation completed with workflow failures"
        for metric_id in metric_ids:
            failures[metric_id] = error_message
    return failures


def _score_outcome(score: Score) -> Literal["correct", "incorrect"]:
    return "correct" if float(score.value) >= 1.0 else "incorrect"


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
            candidate_id=getattr(event, "candidate_id", None),
            metric_id=getattr(event, "metric_id", None),
        )
    )
    return view.model_copy(update={"entries": entries})


def _apply_event_to_trace_view(
    snapshot: RunSnapshot, view: TraceView, event: RunEvent
) -> TraceView:
    generation_traces = list(view.generation_traces)
    conversation_traces = list(view.conversation_traces)
    evaluation_traces = list(view.evaluation_traces)

    if isinstance(event, GenerationCompletedEvent) and event.result is not None:
        result = GenerationResult.model_validate(event.result)
        if result.trace:
            generation_traces.append(
                GenerationTraceRecord(
                    case_id=event.case_id,
                    candidate_id=event.candidate_id,
                    trace_id=f"{event.candidate_id}:generation",
                    steps=[step.model_dump(mode="json") for step in result.trace],
                )
            )
        if result.conversation:
            conversation_traces.append(
                ConversationTraceRecord(
                    case_id=event.case_id,
                    candidate_id=event.candidate_id,
                    trace_id=f"{event.candidate_id}:conversation",
                    messages=[
                        message.model_dump(mode="json")
                        for message in result.conversation
                    ],
                )
            )
    elif isinstance(event, EvaluationCompletedEvent) and event.execution is not None:
        evaluation_traces.append(
            EvaluationTraceRecord(
                case_id=event.case_id,
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
        }
    )
