"""Runtime result, work-item, and resume state models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field

from themis.core.base import FrozenModel, JSONValue
from themis.core.case_refs import resolve_case_key
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
from themis.core.models import (
    Case,
    MetricResult,
    ParsedOutput,
    ReducedCandidate,
    ScoreError,
    Candidate,
)
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

    generated_candidates: dict[str, Candidate] = Field(default_factory=dict)
    generated_candidates_by_index: dict[int, Candidate] = Field(default_factory=dict)
    generated_candidate_blob_refs: dict[str, str] = Field(default_factory=dict)
    generation_failures: dict[str, str] = Field(default_factory=dict)
    generation_failure_keys_by_index: dict[int, str] = Field(default_factory=dict)
    selected_candidate_ids: list[str] | None = None
    selection_metadata: dict[str, object] = Field(default_factory=dict)
    selection_error: str | None = None
    reduced_candidate: ReducedCandidate | None = None
    reduction_error: str | None = None
    parsed_views: dict[str, ParsedOutput] = Field(default_factory=dict)
    parse_errors: dict[str, str] = Field(default_factory=dict)
    evaluation_executions: dict[str, EvaluationExecution] = Field(default_factory=dict)
    evaluation_execution_blob_refs: dict[str, str] = Field(default_factory=dict)
    evaluation_failures: dict[str, str] = Field(default_factory=dict)
    metric_results: dict[str, MetricResult] = Field(default_factory=dict)
    score_failures: dict[str, ScoreError] = Field(default_factory=dict)

    @property
    def scores(self) -> dict[str, MetricResult | ScoreError]:
        return {
            **self.score_failures,
            **self.metric_results,
        }


class ExecutionState(FrozenModel):
    """Persisted run state rebuilt from the run event stream."""

    run_id: str
    status: RunStatus = RunStatus.PENDING
    completed_through_stage: str | None = None
    case_states: dict[str, CaseExecutionState] = Field(default_factory=dict)

    @classmethod
    def from_events(cls, run_id: str, events: list[RunEvent]) -> ExecutionState:
        state = cls(run_id=run_id)
        for event in events:
            state = state.apply_event(event)
        return state

    def apply_event(self, event: RunEvent) -> ExecutionState:
        saw_failures = self.status in {
            RunStatus.FAILED,
            RunStatus.PARTIAL_FAILURE,
        } or any(
            _case_state_has_failures(case_state)
            for case_state in self.case_states.values()
        )

        if isinstance(event, RunStartedEvent):
            return self.model_copy(update={"status": RunStatus.RUNNING})
        if isinstance(event, RunCompletedEvent):
            status = (
                RunStatus.COMPLETED if not saw_failures else RunStatus.PARTIAL_FAILURE
            )
            return self.model_copy(
                update={
                    "status": status,
                    "completed_through_stage": event.completed_through_stage,
                }
            )
        if isinstance(event, RunFailedEvent):
            return self.model_copy(update={"status": RunStatus.FAILED})
        if not isinstance(event, CaseStageEvent):
            return self

        case_key = resolve_case_key(
            case_id=event.case_id,
            dataset_id=getattr(event, "dataset_id", None),
            case_key=getattr(event, "case_key", None),
        )
        case_states = dict(self.case_states)
        legacy_case_alias = (
            event.case_id
            if getattr(event, "dataset_id", None) is None
            and getattr(event, "case_key", None) is None
            else None
        )
        current = case_states.get(case_key)
        if current is None and legacy_case_alias is not None:
            current = case_states.get(legacy_case_alias)
        if current is None:
            current = CaseExecutionState()
        updated = current

        if isinstance(event, GenerationCompletedEvent) and event.result is not None:
            generated = dict(current.generated_candidates)
            generated[event.candidate_id] = Candidate.model_validate(event.result)
            generated_by_index = dict(current.generated_candidates_by_index)
            generated_blob_refs = dict(current.generated_candidate_blob_refs)
            failures = dict(current.generation_failures)
            failure_keys_by_index = dict(current.generation_failure_keys_by_index)
            if event.candidate_index is not None:
                generated_by_index[event.candidate_index] = generated[
                    event.candidate_id
                ]
                failure_key = failure_keys_by_index.pop(event.candidate_index, None)
                if failure_key is not None:
                    failures.pop(failure_key, None)
            failures.pop(event.candidate_id, None)
            if event.result_blob_ref is not None:
                generated_blob_refs[event.candidate_id] = event.result_blob_ref
            updated = current.model_copy(
                update={
                    "generated_candidates": generated,
                    "generated_candidates_by_index": generated_by_index,
                    "generated_candidate_blob_refs": generated_blob_refs,
                    "generation_failures": failures,
                    "generation_failure_keys_by_index": failure_keys_by_index,
                }
            )
        elif isinstance(event, GenerationFailedEvent):
            failures = dict(current.generation_failures)
            failure_keys_by_index = dict(current.generation_failure_keys_by_index)
            failures[event.candidate_id] = event.error_message
            if event.candidate_index is not None:
                failure_keys_by_index[event.candidate_index] = event.candidate_id
            updated = current.model_copy(
                update={
                    "generation_failures": failures,
                    "generation_failure_keys_by_index": failure_keys_by_index,
                }
            )
        elif isinstance(event, SelectionCompletedEvent):
            updated = current.model_copy(
                update={
                    "selected_candidate_ids": list(event.candidate_ids),
                    "selection_metadata": dict(event.metadata),
                    "selection_error": None,
                }
            )
        elif isinstance(event, SelectionFailedEvent):
            updated = current.model_copy(
                update={"selection_error": event.error_message}
            )
        elif isinstance(event, ReductionCompletedEvent) and event.result is not None:
            updated = current.model_copy(
                update={
                    "reduced_candidate": ReducedCandidate.model_validate(event.result)
                }
            )
        elif isinstance(event, ReductionFailedEvent):
            updated = current.model_copy(
                update={"reduction_error": event.error_message}
            )
        elif isinstance(event, ParseCompletedEvent) and event.result is not None:
            parsed_views = dict(current.parsed_views)
            parse_errors = dict(current.parse_errors)
            parsed_views[event.parser_id] = ParsedOutput.model_validate(event.result)
            parse_errors.pop(event.parser_id, None)
            updated = current.model_copy(
                update={"parsed_views": parsed_views, "parse_errors": parse_errors}
            )
        elif isinstance(event, ParseFailedEvent):
            parse_errors = dict(current.parse_errors)
            parse_errors[event.parser_id] = event.error_message
            updated = current.model_copy(update={"parse_errors": parse_errors})
        elif (
            isinstance(event, EvaluationCompletedEvent) and event.execution is not None
        ):
            evaluation_executions = dict(current.evaluation_executions)
            evaluation_execution_blob_refs = dict(
                current.evaluation_execution_blob_refs
            )
            evaluation_failures = dict(current.evaluation_failures)
            evaluation_executions[event.metric_id] = EvaluationExecution.model_validate(
                event.execution
            )
            if event.execution_blob_ref is not None:
                evaluation_execution_blob_refs[event.metric_id] = (
                    event.execution_blob_ref
                )
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
            evaluation_execution_blob_refs = dict(
                current.evaluation_execution_blob_refs
            )
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
        elif isinstance(event, ScoreCompletedEvent) and event.metric_result is not None:
            metric_results = dict(current.metric_results)
            score_failures = dict(current.score_failures)
            metric_results[event.metric_id] = MetricResult.model_validate(
                event.metric_result
            )
            score_failures.pop(event.metric_id, None)
            updated = current.model_copy(
                update={
                    "metric_results": metric_results,
                    "score_failures": score_failures,
                }
            )
        elif isinstance(event, ScoreFailedEvent) and event.error is not None:
            metric_results = dict(current.metric_results)
            score_failures = dict(current.score_failures)
            metric_results.pop(event.metric_id, None)
            score_failures[event.metric_id] = ScoreError.model_validate(event.error)
            updated = current.model_copy(
                update={
                    "metric_results": metric_results,
                    "score_failures": score_failures,
                }
            )

        if legacy_case_alias is not None and legacy_case_alias != case_key:
            case_states.pop(legacy_case_alias, None)
        case_states[case_key] = updated
        status = self.status
        if status is RunStatus.COMPLETED and _case_state_has_failures(updated):
            status = RunStatus.PARTIAL_FAILURE
        return self.model_copy(update={"status": status, "case_states": case_states})


class ExecutionCheckpoint(FrozenModel):
    """Store-level checkpoint for fast resume state lookup."""

    schema_version: str = "2"
    run_id: str
    attempt_id: str | None = None
    event_count: int
    execution_state: ExecutionState
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProjectionCursor(FrozenModel):
    """Projection progress marker over a run event stream."""

    schema_version: str = "2"
    run_id: str
    projection_name: str
    event_count: int
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _case_state_has_failures(case_state: CaseExecutionState) -> bool:
    return any(
        (
            case_state.generation_failures,
            case_state.selection_error is not None,
            case_state.reduction_error is not None,
            case_state.parse_errors,
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
    case_key: str
    candidate_index: int
    candidate_id: str
    seed: int | None = None


class CaseResult(FrozenModel):
    """Final case-level result returned from a run."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    generated_candidates: list[Candidate] = Field(default_factory=list)
    generated_candidate_blob_refs: dict[str, str] = Field(default_factory=dict)
    generation_failures: dict[str, str] = Field(default_factory=dict)
    reduced_candidate: ReducedCandidate | None = None
    reduction_error: str | None = None
    parsed_views: dict[str, ParsedOutput] = Field(default_factory=dict)
    parse_errors: dict[str, str] = Field(default_factory=dict)
    evaluation_executions: list[EvaluationExecution] = Field(default_factory=list)
    evaluation_execution_blob_refs: dict[str, str] = Field(default_factory=dict)
    evaluation_failures: dict[str, str] = Field(default_factory=dict)
    metric_results: list[MetricResult | ScoreError] = Field(default_factory=list)


class RunResult(FrozenModel):
    """Final run-level result returned from execution."""

    run_id: str
    status: RunStatus
    completed_through_stage: str | None = None
    progress: ProgressSnapshot = Field(default_factory=ProgressSnapshot)
    cases: list[CaseResult] = Field(default_factory=list)


class ExecutionResourcePlan(FrozenModel):
    """Resource-facing plan derived from a compiled run snapshot."""

    run_id: str
    estimated_generation_calls: int
    estimated_judge_calls: int
    planned_parse_tasks: int
    planned_score_tasks: int
    required_execution_backends: list[str] = Field(default_factory=list)
    provider_call_counts: dict[str, int] = Field(default_factory=dict)
    stage_parallelism: dict[str, int] = Field(default_factory=dict)
    provider_parallelism: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    assumptions: dict[str, JSONValue] = Field(default_factory=dict)


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
    estimated_generation_input_tokens: int = 0
    estimated_generation_output_tokens: int = 0
    estimated_judge_prompt_tokens: int = 0
    estimated_judge_output_tokens: int = 0
    estimated_total_tokens: int = 0
    resource_plan: ExecutionResourcePlan | None = None
    assumptions: dict[str, JSONValue] = Field(default_factory=dict)


class RerunSelector(FrozenModel):
    """Target subset for a stored-run rerun."""

    failed_only: bool = False
    case_ids: list[str] = Field(default_factory=list)
    case_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    metric_ids: list[str] = Field(default_factory=list)


class RerunPlan(FrozenModel):
    """Runtime rerun request over an existing compiled run."""

    stage: str
    selector: RerunSelector = Field(default_factory=RerunSelector)


class ReductionBundleRecord(FrozenModel):
    """One portable reduction artifact record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    result: ReducedCandidate


class ReductionBundle(FrozenModel):
    """Portable bundle of reduction artifacts for a run."""

    schema_version: str = "1"
    run_id: str
    snapshot: RunSnapshot
    records: list[ReductionBundleRecord] = Field(default_factory=list)


class ParseBundleRecord(FrozenModel):
    """One portable parse artifact record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    result: ParsedOutput


class ParseBundle(FrozenModel):
    """Portable bundle of parse artifacts for a run."""

    schema_version: str = "1"
    run_id: str
    snapshot: RunSnapshot
    records: list[ParseBundleRecord] = Field(default_factory=list)


class ScoreBundleRecord(FrozenModel):
    """One portable score artifact record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    metric_id: str
    metric_result: MetricResult


class ScoreBundle(FrozenModel):
    """Portable bundle of score artifacts for a run."""

    schema_version: str = "1"
    run_id: str
    snapshot: RunSnapshot
    records: list[ScoreBundleRecord] = Field(default_factory=list)


class GenerationBundleRecord(FrozenModel):
    """One portable generation artifact record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    candidate_index: int | None = None
    seed: int | None = None
    result_blob_ref: str | None = None
    result: Candidate


class GenerationBundle(FrozenModel):
    """Portable bundle of generation artifacts for a run."""

    schema_version: str = "1"
    run_id: str
    snapshot: RunSnapshot
    records: list[GenerationBundleRecord] = Field(default_factory=list)


class EvaluationBundleRecord(FrozenModel):
    """One portable evaluation execution record."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
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
