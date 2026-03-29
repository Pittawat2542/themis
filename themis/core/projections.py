"""Projection builders for Phase 4 read models."""

from __future__ import annotations

from collections import defaultdict

from themis.core.base import JSONValue
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    RunEvent,
)
from themis.core.models import GenerationResult
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


def build_run_result(snapshot: RunSnapshot, events: list[RunEvent]) -> RunResult:
    state = ExecutionState.from_events(snapshot.run_id, events)
    case_results: list[CaseResult] = []
    failed_cases = 0

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
                    case_state.score_failures,
                )
            )
            failed_cases += int(has_failure)
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    generated_candidates=[
                        case_state.generated_candidates_by_index[index]
                        for index in sorted(case_state.generated_candidates_by_index)
                    ]
                    or list(case_state.generated_candidates.values()),
                    generated_candidate_blob_refs=dict(case_state.generated_candidate_blob_refs),
                    generation_failures=dict(case_state.generation_failures),
                    reduced_candidate=case_state.reduced_candidate,
                    reduction_error=case_state.reduction_error,
                    parsed_output=case_state.parsed_output,
                    parse_error=case_state.parse_error,
                    evaluation_executions=list(case_state.evaluation_executions.values()),
                    evaluation_execution_blob_refs=dict(case_state.evaluation_execution_blob_refs),
                    evaluation_failures=dict(case_state.evaluation_failures),
                    scores=list(case_state.successful_scores.values()) + list(case_state.score_failures.values()),
                )
            )

    total_cases = sum(len(dataset.cases) for dataset in snapshot.datasets)
    completed_cases = len(case_results) - failed_cases

    return RunResult(
        run_id=snapshot.run_id,
        status=state.status,
        progress=ProgressSnapshot(
            total_cases=total_cases,
            completed_cases=completed_cases,
            failed_cases=failed_cases,
        ),
        cases=case_results,
    )


def build_benchmark_result(snapshot: RunSnapshot, events: list[RunEvent]) -> BenchmarkResult:
    run_result = build_run_result(snapshot, events)
    score_rows: list[BenchmarkScoreRow] = []
    metric_scores: dict[str, list[float]] = defaultdict(list)

    for case in run_result.cases:
        for score in case.scores:
            if not hasattr(score, "value"):
                continue
            row = BenchmarkScoreRow(
                case_id=case.case_id,
                metric_id=score.metric_id,
                value=float(score.value),
                candidate_id=case.reduced_candidate.candidate_id if case.reduced_candidate is not None else None,
            )
            score_rows.append(row)
            metric_scores[row.metric_id].append(row.value)

    metric_ids = sorted(metric_scores)
    metric_means = {
        metric_id: sum(values) / len(values)
        for metric_id, values in metric_scores.items()
        if values
    }

    return BenchmarkResult(
        run_id=snapshot.run_id,
        dataset_ids=[dataset.dataset_id for dataset in snapshot.datasets],
        metric_ids=metric_ids,
        total_cases=run_result.progress.total_cases,
        completed_cases=run_result.progress.completed_cases,
        failed_cases=run_result.progress.failed_cases,
        score_rows=score_rows,
        metric_means=metric_means,
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
    generation_traces: list[GenerationTraceRecord] = []
    conversation_traces: list[ConversationTraceRecord] = []
    evaluation_traces: list[EvaluationTraceRecord] = []

    for event in events:
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
                        messages=[message.model_dump(mode="json") for message in result.conversation],
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

    return TraceView(
        run_id=snapshot.run_id,
        generation_traces=generation_traces,
        conversation_traces=conversation_traces,
        evaluation_traces=evaluation_traces,
    )


def build_projection_payloads(snapshot: RunSnapshot, events: list[RunEvent]) -> dict[str, JSONValue]:
    return {
        "snapshot": snapshot.model_dump(mode="json"),
        "run_result": build_run_result(snapshot, events).model_dump(mode="json"),
        "benchmark_result": build_benchmark_result(snapshot, events).model_dump(mode="json"),
        "timeline_view": build_timeline_view(snapshot, events).model_dump(mode="json"),
        "trace_view": build_trace_view(snapshot, events).model_dump(mode="json"),
    }
