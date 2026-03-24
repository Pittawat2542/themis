"""Projection refresh logic triggered by terminal trial events."""

from __future__ import annotations

from typing import Any, cast

from themis.contracts.protocols import (
    ProjectionRefreshRepository,
    TrialEventRepository,
)
from themis.orchestration.overlay_policy import OverlayRefreshPolicy
from themis.orchestration.trace_scoring import score_trial_traces
from themis.overlays import OverlaySelection
from themis.records.trial import TrialRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.types.enums import RecordStatus
from themis.types.events import (
    ProjectionCompletedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventType,
)


class ProjectionHandler:
    """Materializes trial projections when a terminal trial event is observed."""

    def __init__(
        self,
        event_repo: TrialEventRepository,
        projection_repo: ProjectionRefreshRepository,
        projection_version: str = "v1",
        refresh_policy: OverlayRefreshPolicy | None = None,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.event_repo = event_repo
        self.projection_repo = projection_repo
        self.projection_version = projection_version
        self.refresh_policy = refresh_policy or OverlayRefreshPolicy()
        self.registry = registry
        if self.registry is not None:
            from themis.catalog.runtime.registry import register_catalog_metrics

            register_catalog_metrics(self.registry)

    def on_trial_completed(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord:
        """Materialize or refresh projections for one completed trial overlay."""
        existing_events = self.event_repo.get_events(trial_hash)
        selection = OverlaySelection(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        needs_refresh = self.refresh_policy.needs_refresh(
            existing_events,
            selection=selection,
        )
        if not needs_refresh:
            record = self.projection_repo.materialize_trial_record(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )
            self._require_trace_registry(record)
            return record

        next_seq = (self.event_repo.last_event_index(trial_hash) or 0) + 1
        source_range = (
            (existing_events[0].event_seq, existing_events[-1].event_seq)
            if existing_events
            else None
        )
        projection_event = TrialEvent(
            trial_hash=trial_hash,
            event_seq=next_seq,
            event_id=f"{trial_hash}:{next_seq}",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            status=RecordStatus.OK,
            metadata=ProjectionCompletedEventMetadata(
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
                projection_version=self.projection_version,
                source_event_range=list(source_range)
                if source_range is not None
                else None,
            ),
        )

        shared_manager = getattr(self.event_repo, "manager", None)
        if shared_manager is not None and shared_manager is getattr(
            self.projection_repo, "manager", None
        ):
            with shared_manager.get_connection() as conn:
                with conn:
                    record = self._materialize(
                        trial_hash,
                        transform_hash=transform_hash,
                        evaluation_hash=evaluation_hash,
                        projection_event=projection_event,
                        conn=conn,
                    )
                    cast(Any, self.event_repo).append_event(projection_event, conn=conn)
                    return record

        record = self._materialize(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
            projection_event=projection_event,
        )
        self.event_repo.append_event(projection_event)
        return record

    def _materialize(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None,
        evaluation_hash: str | None,
        projection_event: TrialEvent,
        conn=None,
    ) -> TrialRecord:
        if conn is not None:
            record = cast(Any, self.projection_repo).materialize_trial_record(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
                extra_events=[projection_event],
                conn=conn,
            )
        else:
            record = self.projection_repo.materialize_trial_record(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
                extra_events=[projection_event],
            )
        self._require_trace_registry(record)
        if self.registry is not None:
            trace_rows = score_trial_traces(
                projection_repo=self.projection_repo,
                registry=self.registry,
                record=record,
                trial_hash=trial_hash,
                evaluation_hash=evaluation_hash,
            )
            cast(Any, self.projection_repo).replace_trace_metric_scores(
                trial_hash,
                trace_rows,
                evaluation_hash=evaluation_hash,
            )
        return record

    def _require_trace_registry(self, record: TrialRecord) -> None:
        trial_spec = record.trial_spec
        if (
            trial_spec is not None
            and trial_spec.task.trace_evaluations
            and self.registry is None
        ):
            raise ValueError(
                "ProjectionHandler requires a registry when task trace evaluations "
                "are configured."
            )
