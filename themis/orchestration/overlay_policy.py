"""Typed policy for deciding when one overlay projection must refresh."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from themis.overlays import OverlaySelection
from themis.types.events import (
    CandidateFailureEventMetadata,
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    ProjectionCompletedEventMetadata,
    TrialEvent,
    TrialEventType,
)


@dataclass(frozen=True, slots=True)
class OverlayRefreshPolicy:
    """Encapsulates overlay-specific projection refresh rules."""

    def needs_refresh(
        self,
        events: Sequence[TrialEvent],
        *,
        selection: OverlaySelection,
    ) -> bool:
        matching_projection_events = [
            event
            for event in events
            if event.event_type == TrialEventType.PROJECTION_COMPLETED
            and isinstance(event.metadata, ProjectionCompletedEventMetadata)
            and event.metadata.transform_hash == selection.transform_hash
            and event.metadata.evaluation_hash == selection.evaluation_hash
        ]
        last_projection_seq = (
            matching_projection_events[-1].event_seq
            if matching_projection_events
            else None
        )
        return last_projection_seq is None or any(
            event.event_seq > last_projection_seq
            and self.event_relevant_to_overlay(event, selection=selection)
            for event in events
        )

    def event_relevant_to_overlay(
        self,
        event: TrialEvent,
        *,
        selection: OverlaySelection,
    ) -> bool:
        if event.event_type == TrialEventType.PROJECTION_COMPLETED:
            return False
        metadata = event.metadata
        if selection.evaluation_hash is not None:
            return (
                isinstance(
                    metadata,
                    (
                        CandidateFailureEventMetadata,
                        EvaluationCompletedEventMetadata,
                        ProjectionCompletedEventMetadata,
                    ),
                )
                and metadata.evaluation_hash == selection.evaluation_hash
            ) or (
                event.stage is None
                and event.event_type
                in {
                    TrialEventType.CANDIDATE_STARTED,
                    TrialEventType.CANDIDATE_COMPLETED,
                    TrialEventType.CANDIDATE_FAILED,
                    TrialEventType.TRIAL_COMPLETED,
                    TrialEventType.TRIAL_FAILED,
                }
            )
        if selection.transform_hash is not None:
            return (
                isinstance(
                    metadata,
                    (
                        CandidateFailureEventMetadata,
                        ExtractionCompletedEventMetadata,
                        EvaluationCompletedEventMetadata,
                        ProjectionCompletedEventMetadata,
                    ),
                )
                and metadata.transform_hash == selection.transform_hash
            ) or (
                event.stage is None
                and event.event_type
                in {
                    TrialEventType.CANDIDATE_STARTED,
                    TrialEventType.CANDIDATE_COMPLETED,
                    TrialEventType.CANDIDATE_FAILED,
                    TrialEventType.TRIAL_COMPLETED,
                    TrialEventType.TRIAL_FAILED,
                }
            )
        return True
