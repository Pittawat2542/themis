"""Internal timeline and view assembly helpers for SQLite projections."""

from __future__ import annotations

from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.runtime.timeline_view import RecordTimelineView
from themis.storage._projection_codec import ProjectionCodecs
from themis.storage._projection_overlay import ProjectionOverlayReader
from themis.storage._protocols import StorageConnectionManager
from themis.storage.projection_materializer import ProjectionMaterializer
from themis.storage.projection_queries import ProjectionQueries
from themis.types.enums import RecordType


class ProjectionTimelineViews:
    """Builds persisted timelines and rich timeline views from projections."""

    def __init__(
        self,
        manager: StorageConnectionManager,
        queries: ProjectionQueries,
        materializer: ProjectionMaterializer,
        overlay_reader: ProjectionOverlayReader,
        codecs: ProjectionCodecs,
    ) -> None:
        self.manager = manager
        self.queries = queries
        self.materializer = materializer
        self.overlay_reader = overlay_reader
        self.codecs = codecs

    def get_record_timeline(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimeline | None:
        resolved_record_type = RecordType(record_type)
        overlay_key = self.queries.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        trial_hash = self._resolve_trial_hash(
            record_id,
            resolved_record_type,
            overlay_key,
        )
        if trial_hash is None:
            return None
        if (
            self._materialize_trial_record(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
                overlay_key=overlay_key,
            )
            is None
        ):
            return None
        return self._load_record_timeline(record_id, resolved_record_type, overlay_key)

    def get_timeline_view(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        resolved_record_type = RecordType(record_type)
        overlay_key = self.queries.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        trial_hash = self._resolve_trial_hash(
            record_id,
            resolved_record_type,
            overlay_key,
        )
        if trial_hash is None:
            return None
        trial = self._materialize_trial_record(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        if trial is None or trial.trial_spec is None:
            return None
        timeline = self._load_record_timeline(
            record_id,
            resolved_record_type,
            overlay_key,
        )
        if timeline is None:
            return None
        if resolved_record_type == RecordType.TRIAL:
            return RecordTimelineView(
                record_id=record_id,
                record_type=RecordType.TRIAL,
                trial_hash=trial.spec_hash,
                lineage=self.overlay_reader.lineage_for_trial(
                    trial.trial_spec,
                    transform_hash=transform_hash,
                    evaluation_hash=evaluation_hash,
                ),
                trial_spec=trial.trial_spec,
                item_payload=self.overlay_reader.item_payload_for_trial(
                    trial.spec_hash
                ),
                timeline=timeline,
                observability=self.overlay_reader.observability_for_view(
                    trial.spec_hash,
                    None,
                    overlay_key,
                ),
                extractions=[],
                related_events=self.overlay_reader.events_for_overlay(
                    trial.spec_hash,
                    transform_hash=transform_hash,
                    evaluation_hash=evaluation_hash,
                ),
            )

        candidate = next(
            (
                candidate
                for candidate in trial.candidates
                if candidate.spec_hash == record_id
            ),
            None,
        )
        if candidate is None:
            return None
        return RecordTimelineView(
            record_id=record_id,
            record_type=RecordType.CANDIDATE,
            trial_hash=trial_hash,
            candidate_id=record_id,
            lineage=self.overlay_reader.lineage_for_trial(
                trial.trial_spec,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            ),
            trial_spec=trial.trial_spec,
            item_payload=self.overlay_reader.item_payload_for_trial(trial_hash),
            timeline=timeline,
            conversation=candidate.conversation,
            inference=candidate.inference,
            effective_seed=candidate.effective_seed,
            effective_inference_params_hash=candidate.effective_inference_params_hash,
            extractions=candidate.extractions,
            evaluation=candidate.evaluation,
            judge_audit=self.codecs.load_judge_audit(candidate.judge_audits),
            observability=self.overlay_reader.observability_for_view(
                trial_hash, record_id, overlay_key
            ),
            related_events=[
                event
                for event in self.overlay_reader.events_for_overlay(
                    trial_hash,
                    transform_hash=transform_hash,
                    evaluation_hash=evaluation_hash,
                )
                if event.candidate_id == record_id
            ],
        )

    def _resolve_trial_hash(
        self,
        record_id: str,
        record_type: RecordType,
        overlay_key: str,
    ) -> str | None:
        if record_type == RecordType.TRIAL:
            return record_id
        return self.queries.lookup_trial_hash(record_id, overlay_key)

    def _materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        overlay_key: str | None = None,
    ) -> TrialRecord | None:
        resolved_overlay_key = overlay_key or self.queries.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        if not self.queries.overlay_exists_or_materialized(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
            overlay_key=resolved_overlay_key,
        ):
            return None
        return self.materializer.materialize_trial_record(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def _load_record_timeline(
        self,
        record_id: str,
        record_type: RecordType,
        overlay_key: str,
    ) -> RecordTimeline | None:
        with self.manager.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM record_timeline
                WHERE record_id = ? AND record_type = ? AND overlay_key = ?
                ORDER BY stage_order ASC
                """,
                (record_id, record_type.value, overlay_key),
            ).fetchall()
        if not rows:
            return None

        trial_spec = self.queries.load_trial_spec(rows[0]["trial_hash"])
        stages = [self.codecs.load_timeline_stage_record(row) for row in rows]
        source_start = rows[0]["source_start_seq"]
        source_end = rows[0]["source_end_seq"]
        return RecordTimeline(
            record_id=record_id,
            record_type=record_type,
            trial_hash=rows[0]["trial_hash"],
            candidate_id=rows[0]["candidate_id"],
            item_id=trial_spec.item_id,
            stages=stages,
            source_event_range=(source_start, source_end)
            if source_start is not None and source_end is not None
            else None,
        )
