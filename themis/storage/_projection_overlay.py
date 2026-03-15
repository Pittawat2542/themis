"""Internal overlay filtering and timeline-building helpers for projections."""

from __future__ import annotations

from themis.records.observability import ObservabilitySnapshot
from themis.records.timeline import RecordTimeline, TimelineStageRecord
from themis.specs.experiment import TrialSpec
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.observability import SqliteObservabilityStore
from themis.types.enums import RecordStatus, RecordType
from themis.types.events import (
    ArtifactRole,
    CandidateFailureEventMetadata,
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    InferenceCompletedEventMetadata,
    ItemLoadedEventMetadata,
    ProjectionCompletedEventMetadata,
    PromptRenderedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventType,
)
from themis.types.json_types import JSONDict


class ProjectionOverlayReader:
    """Computes overlay-visible event streams and timeline models."""

    def __init__(
        self,
        event_repo: SqliteEventRepository,
        *,
        artifact_store: ArtifactStore | None = None,
        observability_store: SqliteObservabilityStore | None = None,
    ) -> None:
        self.event_repo = event_repo
        self.artifact_store = artifact_store
        self.observability_store = observability_store

    def events_for_overlay(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events: list[TrialEvent] | None = None,
    ) -> list[TrialEvent]:
        events = self.event_repo.get_events(trial_hash)
        if extra_events:
            events = sorted([*events, *extra_events], key=lambda event: event.event_seq)
        if not events:
            return []

        projection_events = [
            event
            for event in events
            if event.event_type == TrialEventType.PROJECTION_COMPLETED
            and isinstance(event.metadata, ProjectionCompletedEventMetadata)
            and event.metadata.transform_hash == transform_hash
            and event.metadata.evaluation_hash == evaluation_hash
        ]
        if not projection_events:
            visible_events = [
                event
                for event in events
                if self.event_visible_in_overlay(
                    event,
                    transform_hash=transform_hash,
                    evaluation_hash=evaluation_hash,
                )
            ]
            if (
                extra_events is None
                and transform_hash is None
                and evaluation_hash is None
            ):
                return visible_events
            if extra_events is None and any(
                self.event_matches_overlay_stage(
                    event,
                    transform_hash=transform_hash,
                    evaluation_hash=evaluation_hash,
                )
                for event in events
            ):
                return visible_events
            return []

        target_projection = projection_events[-1]
        return [
            event
            for event in events
            if event.event_seq <= target_projection.event_seq
            and self.event_visible_in_overlay(
                event,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )
        ]

    def event_visible_in_overlay(
        self,
        event: TrialEvent,
        *,
        transform_hash: str | None,
        evaluation_hash: str | None,
    ) -> bool:
        if event.event_type == TrialEventType.PROJECTION_COMPLETED:
            metadata = event.metadata
            return (
                isinstance(metadata, ProjectionCompletedEventMetadata)
                and metadata.transform_hash == transform_hash
                and metadata.evaluation_hash == evaluation_hash
            )
        if event.stage == TimelineStage.EXTRACTION:
            metadata = event.metadata
            return (
                transform_hash is not None
                and isinstance(
                    metadata,
                    (
                        CandidateFailureEventMetadata,
                        ExtractionCompletedEventMetadata,
                        EvaluationCompletedEventMetadata,
                    ),
                )
                and metadata.transform_hash == transform_hash
            )
        if event.stage == TimelineStage.EVALUATION:
            metadata = event.metadata
            return (
                evaluation_hash is not None
                and isinstance(
                    metadata,
                    (
                        CandidateFailureEventMetadata,
                        EvaluationCompletedEventMetadata,
                        ProjectionCompletedEventMetadata,
                    ),
                )
                and metadata.evaluation_hash == evaluation_hash
            )
        return True

    def event_matches_overlay_stage(
        self,
        event: TrialEvent,
        *,
        transform_hash: str | None,
        evaluation_hash: str | None,
    ) -> bool:
        if event.stage == TimelineStage.EXTRACTION:
            metadata = event.metadata
            return (
                isinstance(
                    metadata,
                    (
                        CandidateFailureEventMetadata,
                        ExtractionCompletedEventMetadata,
                        EvaluationCompletedEventMetadata,
                    ),
                )
                and metadata.transform_hash == transform_hash
            )
        if event.stage == TimelineStage.EVALUATION:
            metadata = event.metadata
            return (
                isinstance(
                    metadata,
                    (
                        CandidateFailureEventMetadata,
                        EvaluationCompletedEventMetadata,
                        ProjectionCompletedEventMetadata,
                    ),
                )
                and metadata.evaluation_hash == evaluation_hash
            )
        if event.event_type == TrialEventType.PROJECTION_COMPLETED:
            metadata = event.metadata
            return (
                isinstance(metadata, ProjectionCompletedEventMetadata)
                and metadata.transform_hash == transform_hash
                and metadata.evaluation_hash == evaluation_hash
            )
        return False

    def item_payload_for_trial(self, trial_hash: str) -> JSONDict | None:
        for event in self.event_repo.get_events(trial_hash):
            if event.stage != TimelineStage.ITEM_LOAD:
                continue
            if isinstance(event.payload, dict):
                return event.payload
            if self.artifact_store is not None:
                metadata = event.metadata
                artifact_hash = next(
                    (
                        artifact.artifact_hash
                        for artifact in event.artifact_refs
                        if artifact.role == ArtifactRole.ITEM_PAYLOAD
                    ),
                    None,
                )
                if artifact_hash is None and isinstance(
                    metadata, ItemLoadedEventMetadata
                ):
                    artifact_hash = metadata.item_payload_hash
                if artifact_hash is not None:
                    payload = self.artifact_store.read_json(artifact_hash)
                    if isinstance(payload, dict):
                        return payload
        return None

    def observability_for_view(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
    ) -> ObservabilitySnapshot | None:
        if self.observability_store is None:
            return None
        refs = self.observability_store.get_snapshot(
            trial_hash, candidate_id, overlay_key
        )
        if refs is not None:
            return refs
        if candidate_id is None:
            return None
        return self.observability_store.get_snapshot(trial_hash, None, overlay_key)

    def build_timeline(
        self,
        *,
        record_id: str,
        record_type: RecordType | str,
        trial_hash: str,
        candidate_id: str | None,
        item_id: str,
        events: list[TrialEvent],
    ) -> RecordTimeline:
        resolved_record_type = RecordType(record_type)
        stages = [
            self.timeline_stage_from_event(event)
            for event in events
            if event.stage is not None
        ]
        source_range = (events[0].event_seq, events[-1].event_seq) if events else None
        return RecordTimeline(
            record_id=record_id,
            record_type=resolved_record_type,
            trial_hash=trial_hash,
            candidate_id=candidate_id,
            item_id=item_id,
            stages=stages,
            source_event_range=source_range,
        )

    def timeline_stage_from_event(self, event: TrialEvent) -> TimelineStageRecord:
        stage = event.stage
        if stage is None:
            raise ValueError("Timeline stages require event.stage to be populated.")

        metadata = event.metadata
        component_id: str | None = None
        if event.stage == TimelineStage.INFERENCE:
            if isinstance(metadata, InferenceCompletedEventMetadata):
                if metadata.provider and metadata.model_id:
                    component_id = f"{metadata.provider}/{metadata.model_id}"
                else:
                    component_id = metadata.provider or metadata.model_id
        elif event.stage == TimelineStage.EXTRACTION:
            if isinstance(metadata, ExtractionCompletedEventMetadata):
                component_id = metadata.extractor_id
        elif event.stage == TimelineStage.EVALUATION:
            if isinstance(metadata, EvaluationCompletedEventMetadata):
                component_id = metadata.metric_id
        elif event.stage == TimelineStage.PROMPT_RENDER:
            if isinstance(metadata, PromptRenderedEventMetadata):
                component_id = metadata.prompt_template_id
        elif event.stage == TimelineStage.PROJECTION:
            if isinstance(metadata, ProjectionCompletedEventMetadata):
                component_id = metadata.projection_version

        return TimelineStageRecord(
            stage=stage,
            status=event.status
            or (RecordStatus.ERROR if event.error is not None else RecordStatus.OK),
            component_id=component_id,
            started_at=event.event_ts,
            ended_at=event.event_ts,
            duration_ms=0,
            metadata=metadata.as_dict(),
            artifact_refs=event.artifact_refs,
            error=event.error,
        )

    def lineage_for_trial(
        self,
        trial_spec: TrialSpec,
        *,
        transform_hash: str | None,
        evaluation_hash: str | None,
    ) -> dict[str, str | None]:
        return {
            "task": trial_spec.task.task_id,
            "model": trial_spec.model.model_id,
            "item": trial_spec.item_id,
            "prompt_template": trial_spec.prompt.id,
            "transform_hash": transform_hash,
            "evaluation_hash": evaluation_hash,
        }
