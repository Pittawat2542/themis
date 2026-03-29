"""Versioned event models for the Phase 1 run store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from themis.core.base import HashableModel, JSONValue


def _now_utc() -> datetime:
    return datetime.now(UTC)


class RunEvent(HashableModel):
    schema_version: str = "1"
    event_type: str
    run_id: str
    occurred_at: datetime = Field(default_factory=_now_utc)


class RunStartedEvent(RunEvent):
    event_type: Literal["run_started"] = "run_started"


class RunCompletedEvent(RunEvent):
    event_type: Literal["run_completed"] = "run_completed"


class RunFailedEvent(RunEvent):
    event_type: Literal["run_failed"] = "run_failed"
    error_message: str


class GenerationCompletedEvent(RunEvent):
    event_type: Literal["generation_completed"] = "generation_completed"
    case_id: str
    candidate_id: str


class ReductionCompletedEvent(RunEvent):
    event_type: Literal["reduction_completed"] = "reduction_completed"
    case_id: str
    candidate_id: str


class ParseCompletedEvent(RunEvent):
    event_type: Literal["parse_completed"] = "parse_completed"
    case_id: str
    candidate_id: str


class ScoreCompletedEvent(RunEvent):
    event_type: Literal["score_completed"] = "score_completed"
    case_id: str
    candidate_id: str
    metric_id: str


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


EVENT_TYPES: dict[str, type[RunEvent]] = {
    "run_started": RunStartedEvent,
    "run_completed": RunCompletedEvent,
    "run_failed": RunFailedEvent,
    "generation_completed": GenerationCompletedEvent,
    "reduction_completed": ReductionCompletedEvent,
    "parse_completed": ParseCompletedEvent,
    "score_completed": ScoreCompletedEvent,
    "step_started": StepStartedEvent,
    "step_completed": StepCompletedEvent,
    "step_failed": StepFailedEvent,
}


def event_from_dict(payload: dict[str, Any]) -> RunEvent:
    event_type = payload["event_type"]
    return EVENT_TYPES[event_type].model_validate(payload)
