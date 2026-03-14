from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from themis.records.error import ErrorRecord
from themis.types.enums import RecordStatus, RecordType
from themis.types.events import ArtifactRef, TimelineStage
from themis.types.json_types import JSONValueType


class TimelineStageRecord(BaseModel):
    """Normalized representation of one lifecycle stage in a materialized timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: TimelineStage
    status: RecordStatus
    component_id: str | None = None
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    metadata: dict[str, JSONValueType] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error: ErrorRecord | None = None


class RecordTimeline(BaseModel):
    """Stage-by-stage audit trail materialized from append-only trial events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    record_type: RecordType
    trial_hash: str
    candidate_id: str | None = None
    item_id: str
    stages: list[TimelineStageRecord] = Field(default_factory=list)
    source_event_range: tuple[int, int] | None = None
