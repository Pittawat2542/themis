"""Runtime result, work-item, and resume state models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from themis.core.base import FrozenModel
from themis.core.events import (
    EvaluationCompletedEvent,
    EvaluationFailedEvent,
    GenerationCompletedEvent,
    GenerationFailedEvent,
    SelectionCompletedEvent,
    SelectionFailedEvent,
    ParseCompletedEvent,
    ParseFailedEvent,
    ReductionCompletedEvent,
    ReductionFailedEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
    ScoreFailedEvent,
)
from themis.core.models import Case, GenerationResult, ParsedOutput, ReducedCandidate, Score, ScoreError
from themis.core.snapshot import RunSnapshot
from themis.core.workflows import EvaluationExecution

CaseStageEvent = (
    GenerationCompletedEvent
    | GenerationFailedEvent
    | SelectionCompletedEvent
    | SelectionFailedEvent
    | ReductionCompletedEvent
    | ReductionFailedEvent
    | ParseCompletedEvent
    | ParseFailedEvent
    | EvaluationCompletedEvent
    | EvaluationFailedEvent
    | ScoreCompletedEvent
    | ScoreFailedEvent
)


class RunStatus(StrEnum):
    """User-facing run status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_FAILURE = "partial_failure"


class ProgressSnapshot(FrozenModel):
    """Aggregate case progress for a run."""

    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0


class CaseExecutionState(FrozenModel):
    """Persisted per-case execution state derived from stored events."""

    generated_candidates: dict[str, GenerationResult] = Field(default_factory=dict)
    generated_candidates_by_index: dict[int, GenerationResult] = Field(default_factory=dict)
    generated_candidate_blob_refs: dict[str, str] = Field(default_factory=dict)
    generation_failures: dict[str, str] = Field(default_factory=dict)
    selected_candidate_ids: list[str] | None = None
    selection_metadata: dict[str, object] = Field(default_factory=dict)
    selection_error: str | None = None
    reduced_candidate: ReducedCandidate | None = None
    reduction_error: str | None = None
    parsed_output: ParsedOutput | None = None
    parse_error: str | None = None
    evaluation_executions: dict[str, EvaluationExecution] = Field(default_factory=dict)
    evaluation_execution_blob_refs: dict[str, str] = Field(default_factory=dict)
    evaluation_failures: dict[str, str] = Field(default_factory=dict)
    successful_scores: dict[str, Score] = Field(default_factory=dict)
    score_failures: dict[str, ScoreError] = Field(default_factory=dict)

    @property
    def scores(self) -> dict[str, Score | ScoreError]:
        return {
            **self.score_failures,
            **self.successful_scores,
        }


class ExecutionState(FrozenModel):
    """Persisted run state rebuilt from the run event stream."""

    run_id: str
    status: RunStatus = RunStatus.PENDING
    case_states: dict[str, CaseExecutionState] = Field(default_factory=dict)

    @classmethod
    def from_events(cls, run_id: str, events: list[RunEvent]) -> ExecutionState:
        state = cls(run_id=run_id)
        for event in events:
            state = state.apply_event(event)
        return state

    def apply_event(self, event: RunEvent) -> ExecutionState:
        saw_failures = self.status in {RunStatus.FAILED, RunStatus.PARTIAL_FAILURE} or any(
            _case_state_has_failures(case_state) for case_state in self.case_states.values()
        )

        if isinstance(event, RunStartedEvent):
            return self.model_copy(update={"status": RunStatus.RUNNING})
        if isinstance(event, RunCompletedEvent):
            status = RunStatus.COMPLETED if not saw_failures else RunStatus.PARTIAL_FAILURE
            return self.model_copy(update={"status": status})
        if isinstance(event, RunFailedEvent):
            return self.model_copy(update={"status": RunStatus.FAILED})
        if not isinstance(event, CaseStageEvent):
            return self

        case_states = dict(self.case_states)
        current = case_states.get(event.case_id, CaseExecutionState())
        updated = current

        if isinstance(event, GenerationCompletedEvent) and event.result is not None:
            generated = dict(current.generated_candidates)
            generated[event.candidate_id] = GenerationResult.model_validate(event.result)
            generated_by_index = dict(current.generated_candidates_by_index)
            generated_blob_refs = dict(current.generated_candidate_blob_refs)
            if event.candidate_index is not None:
                generated_by_index[event.candidate_index] = generated[event.candidate_id]
            if event.result_blob_ref is not None:
                generated_blob_refs[event.candidate_id] = event.result_blob_ref
            updated = current.model_copy(
                update={
                    "generated_candidates": generated,
                    "generated_candidates_by_index": generated_by_index,
                    "generated_candidate_blob_refs": generated_blob_refs,
                }
            )
        elif isinstance(event, GenerationFailedEvent):
            failures = dict(current.generation_failures)
            failures[event.candidate_id] = event.error_message
            updated = current.model_copy(update={"generation_failures": failures})
        elif isinstance(event, SelectionCompletedEvent):
            updated = current.model_copy(
                update={
                    "selected_candidate_ids": list(event.candidate_ids),
                    "selection_metadata": dict(event.metadata),
                    "selection_error": None,
                }
            )
        elif isinstance(event, SelectionFailedEvent):
            updated = current.model_copy(update={"selection_error": event.error_message})
        elif isinstance(event, ReductionCompletedEvent) and event.result is not None:
            updated = current.model_copy(
                update={"reduced_candidate": ReducedCandidate.model_validate(event.result)}
            )
        elif isinstance(event, ReductionFailedEvent):
            updated = current.model_copy(update={"reduction_error": event.error_message})
        elif isinstance(event, ParseCompletedEvent) and event.result is not None:
            updated = current.model_copy(update={"parsed_output": ParsedOutput.model_validate(event.result)})
        elif isinstance(event, ParseFailedEvent):
            updated = current.model_copy(update={"parse_error": event.error_message})
        elif isinstance(event, EvaluationCompletedEvent) and event.execution is not None:
            evaluation_executions = dict(current.evaluation_executions)
            evaluation_execution_blob_refs = dict(current.evaluation_execution_blob_refs)
            evaluation_failures = dict(current.evaluation_failures)
            evaluation_executions[event.metric_id] = EvaluationExecution.model_validate(event.execution)
            if event.execution_blob_ref is not None:
                evaluation_execution_blob_refs[event.metric_id] = event.execution_blob_ref
            evaluation_failures.pop(event.metric_id, None)
            updated = current.model_copy(
                update={
                    "evaluation_executions": evaluation_executions,
                    "evaluation_execution_blob_refs": evaluation_execution_blob_refs,
                    "evaluation_failures": evaluation_failures,
                }
            )
        elif isinstance(event, EvaluationFailedEvent):
            evaluation_executions = dict(current.evaluation_executions)
            evaluation_execution_blob_refs = dict(current.evaluation_execution_blob_refs)
            evaluation_failures = dict(current.evaluation_failures)
            evaluation_executions.pop(event.metric_id, None)
            evaluation_execution_blob_refs.pop(event.metric_id, None)
            evaluation_failures[event.metric_id] = event.error_message
            updated = current.model_copy(
                update={
                    "evaluation_executions": evaluation_executions,
                    "evaluation_execution_blob_refs": evaluation_execution_blob_refs,
                    "evaluation_failures": evaluation_failures,
                }
            )
        elif isinstance(event, ScoreCompletedEvent) and event.score is not None:
            successful_scores = dict(current.successful_scores)
            score_failures = dict(current.score_failures)
            successful_scores[event.metric_id] = Score.model_validate(event.score)
            score_failures.pop(event.metric_id, None)
            updated = current.model_copy(
                update={
                    "successful_scores": successful_scores,
                    "score_failures": score_failures,
                }
            )
        elif isinstance(event, ScoreFailedEvent) and event.error is not None:
            successful_scores = dict(current.successful_scores)
            score_failures = dict(current.score_failures)
            successful_scores.pop(event.metric_id, None)
            score_failures[event.metric_id] = ScoreError.model_validate(event.error)
            updated = current.model_copy(
                update={
                    "successful_scores": successful_scores,
                    "score_failures": score_failures,
                }
            )

        case_states[event.case_id] = updated
        status = self.status
        if status is RunStatus.COMPLETED and _case_state_has_failures(updated):
            status = RunStatus.PARTIAL_FAILURE
        return self.model_copy(update={"status": status, "case_states": case_states})


def _case_state_has_failures(case_state: CaseExecutionState) -> bool:
    return any(
        (
            case_state.generation_failures,
            case_state.selection_error is not None,
            case_state.reduction_error is not None,
            case_state.parse_error is not None,
            case_state.evaluation_failures,
            any(
                execution.status == "partial_failure" or bool(execution.failures)
                for execution in case_state.evaluation_executions.values()
            ),
            case_state.score_failures,
        )
    )


class GenerationWorkItem(FrozenModel):
    """Planner output for one generation task."""

    run_id: str
    dataset_id: str
    case: Case
    case_id: str
    candidate_index: int
    candidate_id: str
    seed: int | None = None


class CaseResult(FrozenModel):
    """Final case-level result returned from a run."""

    case_id: str
    generated_candidates: list[GenerationResult] = Field(default_factory=list)
    generated_candidate_blob_refs: dict[str, str] = Field(default_factory=dict)
    generation_failures: dict[str, str] = Field(default_factory=dict)
    reduced_candidate: ReducedCandidate | None = None
    reduction_error: str | None = None
    parsed_output: ParsedOutput | None = None
    parse_error: str | None = None
    evaluation_executions: list[EvaluationExecution] = Field(default_factory=list)
    evaluation_execution_blob_refs: dict[str, str] = Field(default_factory=dict)
    evaluation_failures: dict[str, str] = Field(default_factory=dict)
    scores: list[Score | ScoreError] = Field(default_factory=list)


class RunResult(FrozenModel):
    """Final run-level result returned from execution."""

    run_id: str
    status: RunStatus
    progress: ProgressSnapshot = Field(default_factory=ProgressSnapshot)
    cases: list[CaseResult] = Field(default_factory=list)


class RunEstimate(FrozenModel):
    """Planner estimate for the work implied by a compiled run."""

    run_id: str
    total_cases: int
    candidate_count: int
    metric_count: int
    pure_metric_count: int
    workflow_metric_count: int
    planned_generation_tasks: int
    planned_reduction_tasks: int
    planned_parse_tasks: int
    planned_score_tasks: int


class GenerationBundleRecord(FrozenModel):
    """One portable generation artifact record."""

    case_id: str
    candidate_id: str
    candidate_index: int | None = None
    seed: int | None = None
    result_blob_ref: str | None = None
    result: GenerationResult


class GenerationBundle(FrozenModel):
    """Portable bundle of generation artifacts for a run."""

    schema_version: str = "1"
    run_id: str
    snapshot: RunSnapshot
    records: list[GenerationBundleRecord] = Field(default_factory=list)


class EvaluationBundleRecord(FrozenModel):
    """One portable evaluation execution record."""

    case_id: str
    metric_id: str
    candidate_id: str | None = None
    execution_blob_ref: str | None = None
    execution: EvaluationExecution


class EvaluationBundle(FrozenModel):
    """Portable bundle of evaluation artifacts for a run."""

    schema_version: str = "1"
    run_id: str
    snapshot: RunSnapshot
    records: list[EvaluationBundleRecord] = Field(default_factory=list)
