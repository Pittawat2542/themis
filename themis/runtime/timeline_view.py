from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from themis.records.conversation import Conversation
from themis.records.evaluation import EvaluationRecord
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.observability import ObservabilityRefs
from themis.records.timeline import RecordTimeline
from themis.specs.experiment import TrialSpec
from themis.types.events import TrialEvent
from themis.types.json_types import JSONValueType


class RecordTimelineView(BaseModel):
    """Analysis-oriented single-record projection over timelines and related artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    record_type: Literal["candidate", "trial"]
    trial_hash: str
    candidate_id: str | None = None
    lineage: dict[str, str | None] = Field(default_factory=dict)
    trial_spec: TrialSpec
    item_payload: dict[str, JSONValueType] | None = None
    timeline: RecordTimeline
    conversation: Conversation | None = None
    inference: InferenceRecord | None = None
    extractions: list[ExtractionRecord] = Field(default_factory=list)
    evaluation: EvaluationRecord | None = None
    judge_audit: JudgeAuditTrail | None = None
    observability: ObservabilityRefs | None = None
    related_events: list[TrialEvent] = Field(default_factory=list)
