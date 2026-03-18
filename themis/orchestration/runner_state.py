"""Internal state and replay helpers shared by trial-runner execution paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import threading

from pydantic import TypeAdapter

from themis._replay import CandidateReplayState, ResumeState
from themis.contracts.protocols import DatasetContext
from themis.orchestration.resolved_plugins import ResolvedTrialPlugins
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
    ResolvedTaskStages,
)
from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation, ConversationEvent
from themis.records.provenance import ProvenanceRecord
from themis.specs.experiment import RuntimeContext, TrialSpec
from themis.storage.artifact_store import ArtifactStore
from themis.types.events import ArtifactRef, ArtifactRole, TrialEvent, TrialEventType
from themis.types.json_types import JSONValueType
from themis.types.json_validation import dump_storage_json_bytes, validate_json_value

_CONVERSATION_EVENT_ADAPTER: TypeAdapter[ConversationEvent] = TypeAdapter(
    ConversationEvent
)


@dataclass(slots=True)
class CandidateStageResults:
    """All materialized candidate views produced for one execution attempt."""

    generated_candidate: CandidateRecord
    transformed_candidates: tuple[tuple[ResolvedOutputTransform, CandidateRecord], ...]
    evaluated_candidates: tuple[tuple[ResolvedEvaluation, CandidateRecord], ...]
    final_candidate: CandidateRecord


@dataclass(slots=True)
class TrialExecutionSession:
    """Shared mutable execution state for one trial across candidate workers."""

    trial: TrialSpec
    prepared_trial: TrialSpec
    dataset_context: DatasetContext
    base_runtime: RuntimeContext
    provenance: ProvenanceRecord
    resolved_stages: ResolvedTaskStages
    prompt_payload: JSONValueType
    prompt_artifact: tuple[ArtifactRef, str]
    item_payload: JSONValueType
    dataset_metadata: Mapping[str, object]
    event_seq: int
    resolved_plugins: ResolvedTrialPlugins | None = None
    event_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def trial_hash(self) -> str:
        """Return the deterministic generation identity for the session trial."""
        return self.trial.spec_hash

    def require_resolved_plugins(self) -> ResolvedTrialPlugins:
        """Return the fully resolved stage runtime required for execution."""
        if self.resolved_plugins is None:
            raise RuntimeError(
                "TrialExecutionSession is missing resolved stage runtime."
            )
        return self.resolved_plugins


def json_value(data: object, *, label: str) -> JSONValueType:
    """Validate that a value is safe to persist as structured JSON."""
    return validate_json_value(data, label=label)


def artifact_ref(
    data: object,
    *,
    role: ArtifactRole,
    label: str,
    artifact_store: ArtifactStore | None = None,
) -> tuple[ArtifactRef, str]:
    """Persist or deterministically hash one JSON artifact payload."""
    payload = dump_storage_json_bytes(data, label=label)
    artifact_hash = (
        artifact_store.put_blob(payload, "application/json")
        if artifact_store is not None
        else f"sha256:{hashlib.sha256(payload).hexdigest()}"
    )
    return (
        ArtifactRef(
            artifact_hash=artifact_hash,
            media_type="application/json",
            label=label,
            role=role,
        ),
        artifact_hash,
    )


def dataset_payload(dataset_context: DatasetContext) -> dict[str, object]:
    """Normalize one dataset item into the stored item payload shape."""
    if hasattr(dataset_context, "payload"):
        payload = getattr(dataset_context, "payload")
        if isinstance(payload, dict):
            return dict(payload)
    if isinstance(dataset_context, Mapping):
        return {str(key): value for key, value in dataset_context.items()}
    return {}


def resume_state_from_events(
    candidate_id: str,
    candidate_events: list[TrialEvent],
) -> ResumeState | None:
    """Rebuild resume state from persisted conversation events."""
    if not candidate_events:
        return None

    attempt = 0
    for event in reversed(candidate_events):
        if event.event_type != TrialEventType.TRIAL_RETRY:
            continue
        metadata_attempt = getattr(event.metadata, "attempt", None)
        if isinstance(metadata_attempt, int):
            attempt = metadata_attempt
            break
        payload_attempt = (
            event.payload.get("attempt") if isinstance(event.payload, dict) else None
        )
        if isinstance(payload_attempt, int):
            attempt = payload_attempt
            break

    conversation_events = [
        _CONVERSATION_EVENT_ADAPTER.validate_python(event.payload)
        for event in candidate_events
        if event.event_type == TrialEventType.CONVERSATION_EVENT
        and event.payload is not None
    ]
    if not conversation_events:
        return None

    return ResumeState(
        candidate_id=candidate_id,
        conversation=Conversation(events=conversation_events),
        last_event_index=conversation_events[-1].event_index,
        attempt=attempt,
    )


def candidate_from_terminal_events(
    candidate_id: str,
    sample_index: int,
    candidate_events: list[TrialEvent],
) -> CandidateRecord | None:
    """Replay one candidate from an already-terminal event stream."""
    terminal_events = [
        event
        for event in candidate_events
        if event.event_type
        in {TrialEventType.CANDIDATE_COMPLETED, TrialEventType.CANDIDATE_FAILED}
    ]
    if not terminal_events:
        return None

    state = CandidateReplayState(sample_index=sample_index)
    for event in candidate_events:
        state.apply_event(event)
    return state.to_candidate_record(candidate_id)
