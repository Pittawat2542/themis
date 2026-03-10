"""Projection refresh logic triggered by terminal trial events."""

from __future__ import annotations

from themis.types.enums import RecordStatus
from themis.types.events import TrialEvent, TrialEventType


class ProjectionHandler:
    """Materializes trial projections when a terminal trial event is observed."""

    def __init__(self, event_repo, projection_repo, projection_version: str = "v1"):
        self.event_repo = event_repo
        self.projection_repo = projection_repo
        self.projection_version = projection_version

    def on_trial_completed(self, trial_hash: str, eval_revision: str = "latest"):
        """Materialize or refresh projections for one completed trial."""
        existing_events = self.event_repo.get_events(trial_hash)
        has_projection_event = any(
            event.event_type == TrialEventType.PROJECTION_COMPLETED
            and event.metadata.get("eval_revision") == eval_revision
            for event in existing_events
        )
        if has_projection_event:
            return self.projection_repo.materialize_trial_record(
                trial_hash, eval_revision
            )

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
            stage="projection",
            status=RecordStatus.OK,
            metadata={
                "eval_revision": eval_revision,
                "projection_version": self.projection_version,
                "source_event_range": list(source_range)
                if source_range is not None
                else None,
            },
        )

        shared_manager = getattr(self.event_repo, "manager", None)
        if shared_manager is not None and shared_manager is getattr(
            self.projection_repo, "manager", None
        ):
            with shared_manager.get_connection() as conn:
                with conn:
                    record = self._materialize(
                        trial_hash,
                        eval_revision,
                        projection_event=projection_event,
                        conn=conn,
                    )
                    self.event_repo.append_event(projection_event, conn=conn)
                    return record

        record = self._materialize(
            trial_hash,
            eval_revision,
            projection_event=projection_event,
        )
        self.event_repo.append_event(projection_event)
        return record

    def _materialize(
        self,
        trial_hash: str,
        eval_revision: str,
        *,
        projection_event: TrialEvent,
        conn=None,
    ):
        try:
            if conn is not None:
                return self.projection_repo.materialize_trial_record(
                    trial_hash,
                    eval_revision,
                    extra_events=[projection_event],
                    conn=conn,
                )
            return self.projection_repo.materialize_trial_record(
                trial_hash,
                eval_revision,
                extra_events=[projection_event],
            )
        except TypeError:
            if conn is not None:
                return self.projection_repo.materialize_trial_record(
                    trial_hash, eval_revision
                )
            return self.projection_repo.materialize_trial_record(
                trial_hash, eval_revision
            )
