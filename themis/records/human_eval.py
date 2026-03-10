from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from themis.records.base import RecordBase


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class AnnotationRecord(RecordBase):
    """Human annotation attached to a record outside automated scoring."""

    rater_id: str
    rubric_version: str
    label: str
    notes: str | None = None
    time_spent_s: float | None = None
    created_at: datetime = Field(default_factory=_now_utc)


class AdjudicationRecord(RecordBase):
    """Final adjudicated human label after review."""

    final_label: str
    adjudicator_id: str
    rationale: str | None = None
    created_at: datetime = Field(default_factory=_now_utc)
