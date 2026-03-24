"""Normalized trace views used by trace-level metric implementations."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from themis.runtime.timeline_view import RecordTimelineView


class TraceEvent(BaseModel):
    """Normalized event projected from persisted timeline and conversation data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_key: str
    source_kind: str
    kind: str
    trial_hash: str
    candidate_id: str | None = None
    event_seq: int | None = None
    conversation_event_index: int | None = None
    stage: str | None = None
    timestamp: datetime | None = None
    tool_name: str | None = None
    node_id: str | None = None
    source_metadata: dict[str, object] = Field(default_factory=dict)


class TraceView(BaseModel):
    """Analysis-ready trace view spanning one candidate or one full trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    trace_scope: str
    trial_hash: str
    trial_view: RecordTimelineView
    candidate_view: RecordTimelineView | None = None
    candidate_views: list[RecordTimelineView] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
