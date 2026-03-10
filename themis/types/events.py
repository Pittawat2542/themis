from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from themis.records.error import ErrorRecord
from themis.types.enums import RecordStatus
from themis.types.json_types import JSONDict, JSONValueType

TimelineStage = Literal[
    "item_load", "prompt_render", "inference", "extraction", "evaluation", "projection"
]


class ArtifactRole(str, Enum):
    ITEM_PAYLOAD = "item_payload"
    RENDERED_PROMPT = "rendered_prompt"
    INFERENCE_OUTPUT = "inference_output"
    METRIC_DETAILS = "metric_details"
    JUDGE_AUDIT = "judge_audit"


class TrialEventType(str, Enum):
    TRIAL_STARTED = "trial_started"
    ITEM_LOADED = "item_loaded"
    PROMPT_RENDERED = "prompt_rendered"
    CANDIDATE_STARTED = "candidate_started"
    CONVERSATION_EVENT = "conversation_event"
    INFERENCE_COMPLETED = "inference_completed"
    EXTRACTION_COMPLETED = "extraction_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    CANDIDATE_COMPLETED = "candidate_completed"
    CANDIDATE_FAILED = "candidate_failed"
    TRIAL_RETRY = "trial_retry"
    PROJECTION_COMPLETED = "projection_completed"
    TRIAL_COMPLETED = "trial_completed"
    TRIAL_FAILED = "trial_failed"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactRef(BaseModel):
    """Reference to a deduplicated artifact relevant to an event or timeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_hash: str
    media_type: str
    label: str | None = None
    role: ArtifactRole | None = None


class TrialEvent(BaseModel):
    """Typed append-only lifecycle event persisted by the trial event repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    event_seq: int
    event_id: str
    event_type: TrialEventType
    candidate_id: str | None = None
    stage: TimelineStage | None = None
    status: RecordStatus | None = None
    event_ts: datetime = Field(default_factory=_now_utc)
    metadata: JSONDict = Field(default_factory=dict)
    payload: JSONValueType | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error: ErrorRecord | None = None


class ScoreRow(BaseModel):
    """Flattened candidate-level metric projection row used by stats/reporting code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    candidate_id: str
    metric_id: str
    score: float
    details: JSONDict = Field(default_factory=dict)


class TrialSummaryRow(BaseModel):
    """Flattened trial-level summary row used by reporting and comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    model_id: str | None = None
    task_id: str | None = None
    item_id: str | None = None
    status: RecordStatus
