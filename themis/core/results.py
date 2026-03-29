"""Runtime result, work-item, and resume state models for Phase 2."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from themis.core.base import FrozenModel
from themis.core.events import (
    GenerationCompletedEvent,
    GenerationFailedEvent,
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


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_FAILURE = "partial_failure"


class ProgressSnapshot(FrozenModel):
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0


class CaseExecutionState(FrozenModel):
    generated_candidates: dict[str, GenerationResult] = Field(default_factory=dict)
    generation_failures: dict[str, str] = Field(default_factory=dict)
    reduced_candidate: ReducedCandidate | None = None
    reduction_error: str | None = None
    parsed_output: ParsedOutput | None = None
    parse_error: str | None = None
    scores: dict[str, Score | ScoreError] = Field(default_factory=dict)


class ExecutionState(FrozenModel):
    run_id: str
    status: RunStatus = RunStatus.PENDING
    case_states: dict[str, CaseExecutionState] = Field(default_factory=dict)

    @classmethod
    def from_events(cls, run_id: str, events: list[RunEvent]) -> ExecutionState:
        case_states: dict[str, CaseExecutionState] = {}
        status = RunStatus.PENDING
        saw_failures = False
        for event in events:
            if isinstance(event, RunStartedEvent):
                status = RunStatus.RUNNING
                continue
            if isinstance(event, RunCompletedEvent):
                status = RunStatus.COMPLETED if not saw_failures else RunStatus.PARTIAL_FAILURE
                continue
            if isinstance(event, RunFailedEvent):
                status = RunStatus.FAILED
                continue

            if not hasattr(event, "case_id"):
                continue

            case_id = event.case_id
            current = case_states.get(case_id, CaseExecutionState())
            updated = current

            if isinstance(event, GenerationCompletedEvent) and event.result is not None:
                generated = dict(current.generated_candidates)
                generated[event.candidate_id] = GenerationResult.model_validate(event.result)
                updated = current.model_copy(update={"generated_candidates": generated})
            elif isinstance(event, GenerationFailedEvent):
                failures = dict(current.generation_failures)
                failures[event.candidate_id] = event.error_message
                updated = current.model_copy(update={"generation_failures": failures})
                saw_failures = True
            elif isinstance(event, ReductionCompletedEvent) and event.result is not None:
                updated = current.model_copy(
                    update={"reduced_candidate": ReducedCandidate.model_validate(event.result)}
                )
            elif isinstance(event, ReductionFailedEvent):
                updated = current.model_copy(update={"reduction_error": event.error_message})
                saw_failures = True
            elif isinstance(event, ParseCompletedEvent) and event.result is not None:
                updated = current.model_copy(update={"parsed_output": ParsedOutput.model_validate(event.result)})
            elif isinstance(event, ParseFailedEvent):
                updated = current.model_copy(update={"parse_error": event.error_message})
                saw_failures = True
            elif isinstance(event, ScoreCompletedEvent) and event.score is not None:
                scores = dict(current.scores)
                scores[event.metric_id] = Score.model_validate(event.score)
                updated = current.model_copy(update={"scores": scores})
            elif isinstance(event, ScoreFailedEvent) and event.error is not None:
                scores = dict(current.scores)
                scores[event.metric_id] = ScoreError.model_validate(event.error)
                updated = current.model_copy(update={"scores": scores})
                saw_failures = True

            case_states[case_id] = updated

        return cls(run_id=run_id, status=status, case_states=case_states)


class GenerationWorkItem(FrozenModel):
    run_id: str
    dataset_id: str
    case: Case
    case_id: str
    candidate_index: int
    candidate_id: str
    seed: int | None = None


class CaseResult(FrozenModel):
    case_id: str
    generated_candidates: list[GenerationResult] = Field(default_factory=list)
    reduced_candidate: ReducedCandidate | None = None
    parsed_output: ParsedOutput | None = None
    scores: list[Score | ScoreError] = Field(default_factory=list)


class RunResult(FrozenModel):
    run_id: str
    status: RunStatus
    progress: ProgressSnapshot = Field(default_factory=ProgressSnapshot)
    cases: list[CaseResult] = Field(default_factory=list)


class GenerationBundle(FrozenModel):
    run_id: str
    generated_cases: dict[str, list[GenerationResult]] = Field(default_factory=dict)
