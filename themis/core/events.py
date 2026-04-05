"""Versioned event models for the run store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ConfigDict, Field

from themis.core.base import HashableModel, JSONValue


def _now_utc() -> datetime:
    return datetime.now(UTC)


class RunEvent(HashableModel):
    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    schema_version: str = "1"
    event_type: str
    run_id: str
    occurred_at: datetime = Field(default_factory=_now_utc)


class RunStartedEvent(RunEvent):
    event_type: Literal["run_started"] = "run_started"


class RunCompletedEvent(RunEvent):
    event_type: Literal["run_completed"] = "run_completed"
    completed_through_stage: Literal[
        "generate", "reduce", "parse", "score", "judge"
    ] = "judge"


class RunFailedEvent(RunEvent):
    event_type: Literal["run_failed"] = "run_failed"
    error_message: str


class GenerationCompletedEvent(RunEvent):
    event_type: Literal["generation_completed"] = "generation_completed"
    case_id: str
    candidate_id: str
    candidate_index: int | None = None
    seed: int | None = None
    provider_key: str | None = None
    result: dict[str, JSONValue] | None = None
    result_blob_ref: str | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class GenerationFailedEvent(RunEvent):
    event_type: Literal["generation_failed"] = "generation_failed"
    case_id: str
    candidate_id: str
    error_message: str
    retry_history: list[dict[str, JSONValue]] = Field(default_factory=list)


class SelectionCompletedEvent(RunEvent):
    event_type: Literal["selection_completed"] = "selection_completed"
    case_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SelectionFailedEvent(RunEvent):
    event_type: Literal["selection_failed"] = "selection_failed"
    case_id: str
    error_message: str


class ReductionCompletedEvent(RunEvent):
    event_type: Literal["reduction_completed"] = "reduction_completed"
    case_id: str
    candidate_id: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    result: dict[str, JSONValue] | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class ReductionFailedEvent(RunEvent):
    event_type: Literal["reduction_failed"] = "reduction_failed"
    case_id: str
    error_message: str


class ParseCompletedEvent(RunEvent):
    event_type: Literal["parse_completed"] = "parse_completed"
    case_id: str
    candidate_id: str
    result: dict[str, JSONValue] | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class ParseFailedEvent(RunEvent):
    event_type: Literal["parse_failed"] = "parse_failed"
    case_id: str
    candidate_id: str
    error_message: str


class EvaluationCompletedEvent(RunEvent):
    event_type: Literal["evaluation_completed"] = "evaluation_completed"
    case_id: str
    candidate_id: str | None = None
    metric_id: str
    execution: dict[str, JSONValue] | None = None
    execution_blob_ref: str | None = None


class EvaluationFailedEvent(RunEvent):
    event_type: Literal["evaluation_failed"] = "evaluation_failed"
    case_id: str
    candidate_id: str | None = None
    metric_id: str
    error_message: str


class ScoreCompletedEvent(RunEvent):
    event_type: Literal["score_completed"] = "score_completed"
    case_id: str
    candidate_id: str
    metric_id: str
    score: dict[str, JSONValue] | None = None
    cache_hit: bool = False
    source_run_id: str | None = None


class ScoreFailedEvent(RunEvent):
    event_type: Literal["score_failed"] = "score_failed"
    case_id: str
    candidate_id: str
    metric_id: str
    error: dict[str, JSONValue] | None = None


class StepStartedEvent(RunEvent):
    event_type: Literal["step_started"] = "step_started"
    workflow_id: str
    step_id: str
    step_type: str


class StepCompletedEvent(RunEvent):
    event_type: Literal["step_completed"] = "step_completed"
    workflow_id: str
    step_id: str
    step_type: str
    details: dict[str, JSONValue] = Field(default_factory=dict)


class StepFailedEvent(RunEvent):
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
    event_type = payload["event_type"]
    return EVENT_TYPES[event_type].model_validate(payload)
