"""Internal replay/materialization helpers for storage-backed projections."""

from __future__ import annotations

from themis._replay import CandidateReplayState
from themis.records.trial import TrialRecord
from themis.storage._projection_overlay import ProjectionOverlayReader
from themis.storage._projection_persistence import ProjectionWriter
from themis.storage._protocols import StorageConnection, StorageConnectionManager
from themis.storage.projection_queries import ProjectionQueries
from themis.types.enums import RecordStatus, RecordType
from themis.types.events import TrialEvent, TrialEventType


class ProjectionMaterializer:
    """Replays overlay-visible events and refreshes projection tables."""

    def __init__(
        self,
        manager: StorageConnectionManager,
        overlay_reader: ProjectionOverlayReader,
        writer: ProjectionWriter,
        queries: ProjectionQueries,
    ) -> None:
        self.manager = manager
        self.overlay_reader = overlay_reader
        self.writer = writer
        self.queries = queries

    def materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events: list[TrialEvent] | None = None,
        conn: StorageConnection | None = None,
    ) -> TrialRecord:
        overlay_key = self.queries.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        if conn is None:
            with self.manager.get_connection() as local_conn:
                with local_conn:
                    return self.materialize_trial_record(
                        trial_hash,
                        transform_hash=transform_hash,
                        evaluation_hash=evaluation_hash,
                        extra_events=extra_events,
                        conn=local_conn,
                    )

        trial_spec = self.queries.load_trial_spec(trial_hash)
        events = self.overlay_reader.events_for_overlay(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
            extra_events=extra_events,
        )
        if not events:
            raise ValueError(
                "No events found for "
                f"trial {trial_hash}, transform_hash={transform_hash}, "
                f"evaluation_hash={evaluation_hash}."
            )

        candidate_states: dict[str, CandidateReplayState] = {}
        trial_stage_events: list[TrialEvent] = []
        candidate_stage_events: dict[str, list[TrialEvent]] = {}
        trial_status = RecordStatus.OK
        trial_error = None

        for event in events:
            event = self.overlay_reader.resolve_event_payload(event)
            if event.candidate_id is None:
                if event.stage is not None:
                    trial_stage_events.append(event)
                if event.event_type in {
                    TrialEventType.TRIAL_COMPLETED,
                    TrialEventType.TRIAL_FAILED,
                }:
                    if isinstance(event.payload, dict) and "status" in event.payload:
                        trial_status = RecordStatus(event.payload["status"])
                    elif event.error is not None:
                        trial_status = RecordStatus.ERROR
                    trial_error = event.error
                continue

            state = candidate_states.setdefault(
                event.candidate_id,
                CandidateReplayState(sample_index=len(candidate_states)),
            )
            state.apply_event(event)
            if event.stage is not None:
                candidate_stage_events.setdefault(event.candidate_id, []).append(event)

        candidate_timelines = {
            candidate_id: self.overlay_reader.build_timeline(
                record_id=candidate_id,
                record_type=RecordType.CANDIDATE,
                trial_hash=trial_hash,
                candidate_id=candidate_id,
                item_id=trial_spec.item_id,
                events=stage_events,
            )
            for candidate_id, stage_events in candidate_stage_events.items()
        }
        candidates = [
            state.to_candidate_record(
                candidate_id,
                timeline=candidate_timelines.get(candidate_id),
            )
            for candidate_id, state in sorted(
                candidate_states.items(),
                key=lambda item: (item[1].sample_index, item[0]),
            )
        ]
        if transform_hash is not None or evaluation_hash is not None:
            failed_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.status == RecordStatus.ERROR
                ),
                None,
            )
            if failed_candidate is None:
                trial_status = RecordStatus.OK
                trial_error = None
            else:
                trial_status = RecordStatus.ERROR
                trial_error = failed_candidate.error
        trial_timeline = self.overlay_reader.build_timeline(
            record_id=trial_hash,
            record_type=RecordType.TRIAL,
            trial_hash=trial_hash,
            candidate_id=None,
            item_id=trial_spec.item_id,
            events=trial_stage_events,
        )
        trial_record = TrialRecord(
            spec_hash=trial_hash,
            trial_spec=trial_spec,
            status=trial_status,
            error=trial_error,
            candidates=candidates,
            timeline=trial_timeline,
        )

        self.writer.upsert_trial_summary(conn, trial_record, overlay_key)
        self.writer.delete_trial_metric_scores(conn, trial_hash, overlay_key)
        conn.execute(
            "DELETE FROM candidate_summary WHERE trial_hash = ? AND overlay_key = ?",
            (trial_hash, overlay_key),
        )
        for candidate in candidates:
            self.writer.upsert_candidate_summary(
                conn,
                trial_hash=trial_hash,
                overlay_key=overlay_key,
                sample_index=candidate_states[candidate.spec_hash].sample_index,
                candidate=candidate,
            )
        self.writer.replace_metric_scores(conn, candidates, overlay_key)
        conn.execute(
            "DELETE FROM record_timeline WHERE trial_hash = ? AND overlay_key = ?",
            (trial_hash, overlay_key),
        )
        self.writer.insert_timeline(conn, trial_timeline, overlay_key)
        for candidate_timeline in candidate_timelines.values():
            self.writer.insert_timeline(conn, candidate_timeline, overlay_key)

        return trial_record
