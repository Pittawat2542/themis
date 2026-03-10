from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, TypeAdapter

from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation, ConversationEvent
from themis.records.error import ErrorRecord
from themis.records.evaluation import EvaluationRecord
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.timeline import RecordTimeline
from themis.types.enums import RecordStatus
from themis.types.events import ArtifactRole, TrialEvent, TrialEventType

_CONVERSATION_EVENT_ADAPTER: TypeAdapter[ConversationEvent] = TypeAdapter(
    ConversationEvent
)


class ResumeState(BaseModel):
    """Typed candidate resume context reconstructed from persisted conversation events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    conversation: Conversation
    last_event_index: int

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass(slots=True)
class CandidateReplayState:
    """Mutable replay accumulator for candidate-scoped lifecycle events."""

    sample_index: int
    status: RecordStatus = RecordStatus.OK
    inference: InferenceRecord | None = None
    extractions: list[ExtractionRecord] = field(default_factory=list)
    evaluation: EvaluationRecord | None = None
    error: ErrorRecord | None = None
    conversation_events: list[ConversationEvent] = field(default_factory=list)
    judge_audit_refs: list[str] = field(default_factory=list)

    def apply_event(self, event: TrialEvent) -> None:
        if event.event_type == TrialEventType.CANDIDATE_STARTED and isinstance(
            event.payload, dict
        ):
            sample_index = event.payload.get("sample_index")
            if isinstance(sample_index, int):
                self.sample_index = sample_index
            elif isinstance(sample_index, str) and sample_index.isdigit():
                self.sample_index = int(sample_index)
        if (
            event.event_type == TrialEventType.CONVERSATION_EVENT
            and event.payload is not None
        ):
            self.conversation_events.append(
                _CONVERSATION_EVENT_ADAPTER.validate_python(event.payload)
            )
        if event.stage == "inference" and event.payload is not None:
            self.inference = InferenceRecord.model_validate(event.payload)
        elif event.stage == "extraction" and event.payload is not None:
            self.extractions.append(ExtractionRecord.model_validate(event.payload))
        elif event.stage == "evaluation" and event.payload is not None:
            self.evaluation = EvaluationRecord.model_validate(event.payload)
            self.judge_audit_refs = [
                artifact.artifact_hash
                for artifact in event.artifact_refs
                if artifact.role == ArtifactRole.JUDGE_AUDIT
            ]
        elif event.event_type in {
            TrialEventType.CANDIDATE_COMPLETED,
            TrialEventType.CANDIDATE_FAILED,
        }:
            if isinstance(event.payload, dict) and "status" in event.payload:
                self.status = RecordStatus(event.payload["status"])
            if event.error is not None:
                self.status = RecordStatus.ERROR
                self.error = event.error

    def to_candidate_record(
        self,
        candidate_id: str,
        *,
        timeline: RecordTimeline | None = None,
    ) -> CandidateRecord:
        return CandidateRecord(
            spec_hash=candidate_id,
            candidate_id=candidate_id,
            sample_index=self.sample_index,
            status=self.status,
            error=self.error,
            conversation=Conversation(events=self.conversation_events)
            if self.conversation_events
            else None,
            timeline=timeline,
            inference=self.inference,
            extractions=list(self.extractions),
            evaluation=self.evaluation,
            judge_audits=list(self.judge_audit_refs),
        )
