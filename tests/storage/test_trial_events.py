from __future__ import annotations

from datetime import datetime, timezone

from themis.records.error import ErrorRecord
from themis.types.enums import ErrorCode, ErrorWhere
from themis.types.events import (
    ArtifactRef,
    ArtifactRole,
    InferenceCompletedEventMetadata,
    ProjectionCompletedEventMetadata,
    TrialEvent,
    TrialEventType,
)


def test_trial_event_preserves_typed_metadata_payload_and_artifact_refs() -> None:
    event = TrialEvent(
        trial_hash="trial_hash",
        event_seq=7,
        event_id="evt_7",
        event_type=TrialEventType.INFERENCE_COMPLETED,
        candidate_id="candidate_1",
        stage="inference",  # type: ignore
        event_ts=datetime(2026, 3, 8, tzinfo=timezone.utc),
        metadata={"provider": "openai", "model_id": "gpt-4o-mini"},  # type: ignore
        payload={"raw_text": "42"},
        artifact_refs=[
            ArtifactRef(
                artifact_hash="sha256:abc123",
                media_type="application/json",
                label="inference",
                role=ArtifactRole.INFERENCE_OUTPUT,
            )
        ],
        error=ErrorRecord(
            code=ErrorCode.PROVIDER_TIMEOUT,
            message="timed out",
            retryable=True,
            where=ErrorWhere.INFERENCE,
        ),
    )

    assert event.stage == "inference"
    assert isinstance(event.metadata, InferenceCompletedEventMetadata)
    assert event.metadata.provider == "openai"
    assert event.artifact_refs[0].artifact_hash == "sha256:abc123"
    assert event.artifact_refs[0].role == ArtifactRole.INFERENCE_OUTPUT
    assert event.error is not None
    assert event.error.code == ErrorCode.PROVIDER_TIMEOUT


def test_trial_event_accepts_typed_metadata_models() -> None:
    event = TrialEvent(
        trial_hash="trial_hash",
        event_seq=9,
        event_id="evt_9",
        event_type=TrialEventType.PROJECTION_COMPLETED,
        stage="projection",  # type: ignore
        metadata=ProjectionCompletedEventMetadata(
            transform_hash="tx_1",
            evaluation_hash="eval_1",
            projection_version="v2",
            source_event_range=[1, 8],
        ),
    )

    assert isinstance(event.metadata, ProjectionCompletedEventMetadata)
    assert event.metadata.transform_hash == "tx_1"
    assert event.metadata.as_dict() == {
        "transform_hash": "tx_1",
        "evaluation_hash": "eval_1",
        "projection_version": "v2",
        "source_event_range": [1, 8],
    }
