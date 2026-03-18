"""SQLite materialized read models derived from the trial event log."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from themis.overlays import overlay_key_for
from themis.records.conversation import Conversation
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.runtime.timeline_view import RecordTimelineView
from themis.storage._projection_codec import ProjectionCodecs
from themis.storage._projection_overlay import ProjectionOverlayReader
from themis.storage._protocols import StorageConnection, StorageConnectionManager
from themis.storage._projection_persistence import ProjectionWriter
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.projection_materializer import ProjectionMaterializer
from themis.storage.projection_queries import ProjectionQueries
from themis.storage.timeline_views import ProjectionTimelineViews
from themis.types.enums import RecordType
from themis.types.events import ScoreRow, TrialEvent, TrialSummaryRow


class SqliteProjectionRepository:
    """Materialized read-side repository over the typed trial event log."""

    def __init__(
        self,
        manager: StorageConnectionManager,
        artifact_store: ArtifactStore | None = None,
        observability_store: SqliteObservabilityStore | None = None,
    ):
        self.manager = manager
        self.artifact_store = artifact_store
        self.observability_store = observability_store
        self.event_repo = SqliteEventRepository(manager)
        self._overlay_reader = ProjectionOverlayReader(
            self.event_repo,
            artifact_store=artifact_store,
            observability_store=observability_store,
        )
        self._writer = ProjectionWriter()
        self._codecs = ProjectionCodecs(artifact_store)
        self._queries = ProjectionQueries(manager, self.event_repo, self._codecs)
        self._materializer = ProjectionMaterializer(
            manager,
            self._overlay_reader,
            self._writer,
            self._queries,
        )
        self._timeline_views = ProjectionTimelineViews(
            manager,
            self._queries,
            self._materializer,
            self._overlay_reader,
            self._codecs,
        )

    def save_trial_record(
        self,
        record: TrialRecord,
        conn: StorageConnection | None = None,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> None:
        """Persist a trial record into summary and metric projection tables."""
        overlay_key = overlay_key_for(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        if conn is None:
            with self.manager.get_connection() as local_conn:
                with local_conn:
                    self.save_trial_record(
                        record,
                        conn=local_conn,
                        transform_hash=transform_hash,
                        evaluation_hash=evaluation_hash,
                    )
            return

        self._writer.upsert_trial_summary(conn, record, overlay_key)
        self._writer.delete_trial_metric_scores(conn, record.spec_hash, overlay_key)
        conn.execute(
            "DELETE FROM candidate_summary WHERE trial_hash = ? AND overlay_key = ?",
            (record.spec_hash, overlay_key),
        )
        for index, candidate in enumerate(record.candidates):
            self._writer.upsert_candidate_summary(
                conn,
                trial_hash=record.spec_hash,
                overlay_key=overlay_key,
                sample_index=index,
                candidate=candidate,
            )
        self._writer.replace_metric_scores(conn, record.candidates, overlay_key)

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        """Return a projected trial record for one deterministic overlay."""
        overlay_key = overlay_key_for(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        if not self._queries.overlay_exists_or_materialized(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
            overlay_key=overlay_key,
        ):
            return None
        return self._materializer.materialize_trial_record(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
        """Rebuild the stored conversation stream for one candidate."""
        return self._queries.get_conversation(trial_hash, candidate_id)

    def get_record_timeline(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimeline | None:
        """Return the persisted stage timeline for a trial or candidate record."""
        return self._timeline_views.get_record_timeline(
            record_id,
            record_type,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def get_timeline_view(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        """Return a rich timeline view with lineage, events, and observability."""
        return self._timeline_views.get_timeline_view(
            record_id,
            record_type,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events: list[TrialEvent] | None = None,
        conn: StorageConnection | None = None,
    ) -> TrialRecord:
        """Replay events into a trial record and refresh all projection tables."""
        return self._materializer.materialize_trial_record(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
            extra_events=extra_events,
            conn=conn,
        )

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[ScoreRow]:
        """Yield projected metric scores filtered by trial, metric, and evaluation."""
        yield from self._queries.iter_candidate_scores(
            trial_hashes=trial_hashes,
            metric_id=metric_id,
            evaluation_hash=evaluation_hash,
        )

    def iter_trial_summaries(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[TrialSummaryRow]:
        """Yield projected trial summary rows without hydrating full trial records."""
        yield from self._queries.iter_trial_summaries(
            trial_hashes=trial_hashes,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def has_trial(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        """Return whether a completed projected trial exists for the overlay."""
        return self._queries.has_trial(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
