"""Versioned event models for the run store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ConfigDict, Field

from themis.core.base import HashableModel, JSONValue


def _now_utc() -> datetime:
    return datetime.now(UTC)


class RunEvent(HashableModel):
    """Base event persisted for a compiled run."""

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    schema_version: str = "1"
    event_type: str
    run_id: str
    occurred_at: datetime = Field(default_factory=_now_utc)


class RunStartedEvent(RunEvent):
    """Event emitted when orchestration starts for a run."""

    event_type: Literal["run_started"] = "run_started"


class RunCompletedEvent(RunEvent):
    """Event emitted when orchestration completes successfully."""

    event_type: Literal["run_completed"] = "run_completed"
    completed_through_stage: Literal[
        "generate", "reduce", "parse", "score", "judge"
    ] = "judge"


class RunFailedEvent(RunEvent):
    """Event emitted when orchestration aborts with an unrecoverable error."""

    event_type: Literal["run_failed"] = "run_failed"
    error_message: str


class CaseRunEvent(RunEvent):
    """Base event persisted for a case-scoped execution stage."""

    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None


class GenerationCompletedEvent(CaseRunEvent):
    """Event emitted when candidate generation finishes for a case."""

    event_type: Literal["generation_completed"] = "generation_completed"
    candidate_id: str
    candidate_index: int | None = None
    seed: int | None = None
    provider_key: str | None = None
    result: dict[str, JSONValue] | None = None
    result_blob_ref: str | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class GenerationFailedEvent(CaseRunEvent):
    """Event emitted when candidate generation fails for a case."""

    event_type: Literal["generation_failed"] = "generation_failed"
    candidate_id: str
    candidate_index: int | None = None
    error_message: str
    retry_history: list[dict[str, JSONValue]] = Field(default_factory=list)


class SelectionCompletedEvent(CaseRunEvent):
    """Event emitted when candidate selection succeeds."""

    event_type: Literal["selection_completed"] = "selection_completed"
    candidate_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SelectionFailedEvent(CaseRunEvent):
    """Event emitted when candidate selection fails."""

    event_type: Literal["selection_failed"] = "selection_failed"
    error_message: str


class ReductionCompletedEvent(CaseRunEvent):
    """Event emitted when candidate reduction succeeds."""

    event_type: Literal["reduction_completed"] = "reduction_completed"
    candidate_id: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    result: dict[str, JSONValue] | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class ReductionFailedEvent(CaseRunEvent):
    """Event emitted when candidate reduction fails."""

    event_type: Literal["reduction_failed"] = "reduction_failed"
    error_message: str


class ParseCompletedEvent(CaseRunEvent):
    """Event emitted when parsing a reduced candidate succeeds."""

    event_type: Literal["parse_completed"] = "parse_completed"
    candidate_id: str
    result: dict[str, JSONValue] | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class ParseFailedEvent(CaseRunEvent):
    """Event emitted when parsing a reduced candidate fails."""

    event_type: Literal["parse_failed"] = "parse_failed"
    candidate_id: str
    error_message: str


class EvaluationCompletedEvent(CaseRunEvent):
    """Event emitted when a workflow-backed metric finishes."""

    event_type: Literal["evaluation_completed"] = "evaluation_completed"
    candidate_id: str | None = None
    metric_id: str
    execution: dict[str, JSONValue] | None = None
    execution_blob_ref: str | None = None


class EvaluationFailedEvent(CaseRunEvent):
    """Event emitted when a workflow-backed metric fails."""

    event_type: Literal["evaluation_failed"] = "evaluation_failed"
    candidate_id: str | None = None
    metric_id: str
    error_message: str


class ScoreCompletedEvent(CaseRunEvent):
    """Event emitted when a pure metric succeeds."""

    event_type: Literal["score_completed"] = "score_completed"
    candidate_id: str
    metric_id: str
    score: dict[str, JSONValue] | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class ScoreFailedEvent(CaseRunEvent):
    """Event emitted when a pure metric produces an error payload."""

    event_type: Literal["score_failed"] = "score_failed"
    candidate_id: str
    metric_id: str
    error: dict[str, JSONValue] | None = None


class StepStartedEvent(RunEvent):
    """Event emitted when a workflow step starts."""

    event_type: Literal["step_started"] = "step_started"
    workflow_id: str
    step_id: str
    step_type: str


class StepCompletedEvent(RunEvent):
    """Event emitted when a workflow step completes."""

    event_type: Literal["step_completed"] = "step_completed"
    workflow_id: str
    step_id: str
    step_type: str
    details: dict[str, JSONValue] = Field(default_factory=dict)


class StepFailedEvent(RunEvent):
    """Event emitted when a workflow step fails."""

    event_type: Literal["step_failed"] = "step_failed"
    workflow_id: str
    step_id: str
    step_type: str
    error_message: str
    retry_history: list[dict[str, JSONValue]] = Field(default_factory=list)


EVENT_TYPES: dict[str, type[RunEvent]] = {
    "run_started": RunStartedEvent,
    "run_completed": RunCompletedEvent,
    "run_failed": RunFailedEvent,
    "generation_completed": GenerationCompletedEvent,
    "generation_failed": GenerationFailedEvent,
    "selection_completed": SelectionCompletedEvent,
    "selection_failed": SelectionFailedEvent,
    "reduction_completed": ReductionCompletedEvent,
    "reduction_failed": ReductionFailedEvent,
    "parse_completed": ParseCompletedEvent,
    "parse_failed": ParseFailedEvent,
    "evaluation_completed": EvaluationCompletedEvent,
    "evaluation_failed": EvaluationFailedEvent,
    "score_completed": ScoreCompletedEvent,
    "score_failed": ScoreFailedEvent,
    "step_started": StepStartedEvent,
    "step_completed": StepCompletedEvent,
    "step_failed": StepFailedEvent,
}


def event_from_dict(payload: dict[str, Any]) -> RunEvent:
    """Deserialize a stored event payload into the correct event model."""

    event_type = payload["event_type"]
    return EVENT_TYPES[event_type].model_validate(payload)
