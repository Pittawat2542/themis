"""SQLite materialized read models derived from the trial event log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from themis.errors.exceptions import StorageError
from themis._replay import CandidateReplayState
from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation, ConversationEvent
from themis.records.error import ErrorRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.observability import ObservabilityRefs
from themis.records.timeline import RecordTimeline, TimelineStageRecord
from themis.records.trial import TrialRecord
from themis.runtime.timeline_view import RecordTimelineView
from themis.specs.experiment import TrialSpec
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import (
    ArtifactRef,
    ScoreRow,
    TrialEvent,
    TrialEventType,
    TrialSummaryRow,
)
from themis.types.json_types import JSONDict, JSONList, JSONValueType
from themis.types.json_validation import format_validation_error

_CONVERSATION_EVENT_ADAPTER: TypeAdapter[ConversationEvent] = TypeAdapter(
    ConversationEvent
)
_JSON_DICT_ADAPTER: TypeAdapter[JSONDict] = TypeAdapter(JSONDict)
_JSON_LIST_ADAPTER: TypeAdapter[JSONList] = TypeAdapter(JSONList)
_JSON_VALUE_ADAPTER: TypeAdapter[JSONValueType] = TypeAdapter(JSONValueType)


class SqliteProjectionRepository:
    """Materialized read-side repository over the typed trial event log."""

    def __init__(
        self,
        manager: DatabaseManager,
        artifact_store: ArtifactStore | None = None,
        observability_store: SqliteObservabilityStore | None = None,
    ):
        self.manager = manager
        self.artifact_store = artifact_store
        self.observability_store = observability_store
        self.event_repo = SqliteEventRepository(manager)

    def save_trial_record(
        self,
        record: TrialRecord,
        conn: sqlite3.Connection | None = None,
        *,
        eval_revision: str = "latest",
    ) -> None:
        """Persist a trial record into summary and metric projection tables."""
        if conn is None:
            with self.manager.get_connection() as local_conn:
                with local_conn:
                    self.save_trial_record(
                        record, conn=local_conn, eval_revision=eval_revision
                    )
            return

        self._upsert_trial_summary(conn, record)
        self._delete_trial_metric_scores(conn, record.spec_hash, eval_revision)
        conn.execute(
            "DELETE FROM candidate_summary WHERE trial_hash = ? AND eval_revision = ?",
            (record.spec_hash, eval_revision),
        )
        for index, candidate in enumerate(record.candidates):
            self._upsert_candidate_summary(
                conn,
                trial_hash=record.spec_hash,
                eval_revision=eval_revision,
                sample_index=index,
                candidate=candidate,
            )
        self._replace_metric_scores(conn, record.candidates, eval_revision)

    def get_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord | None:
        """Return a projected trial record when the revision is materialized."""
        if not (
            self.event_repo.has_projection_for_revision(trial_hash, eval_revision)
            or self._has_materialized_trial(trial_hash, eval_revision)
        ):
            return None
        return self.materialize_trial_record(trial_hash, eval_revision)

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
        """Rebuild the stored conversation stream for one candidate."""
        events = [
            _CONVERSATION_EVENT_ADAPTER.validate_python(event.payload)
            for event in self.event_repo.get_events(
                trial_hash, candidate_id=candidate_id
            )
            if event.event_type == TrialEventType.CONVERSATION_EVENT
            and event.payload is not None
        ]
        if not events:
            return None
        return Conversation(events=events)

    def get_record_timeline(
        self,
        record_id: str,
        record_type: Literal["trial", "candidate"],
        eval_revision: str,
    ) -> RecordTimeline | None:
        """Return the persisted stage timeline for a trial or candidate record."""
        trial_hash = (
            record_id
            if record_type == "trial"
            else self._lookup_trial_hash(record_id, eval_revision)
        )
        if trial_hash is None:
            return None
        if not (
            self.event_repo.has_projection_for_revision(trial_hash, eval_revision)
            or self._has_materialized_trial(trial_hash, eval_revision)
        ):
            return None
        self.materialize_trial_record(trial_hash, eval_revision)

        with self.manager.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM record_timeline
                WHERE record_id = ? AND record_type = ? AND eval_revision = ?
                ORDER BY stage_order ASC
                """,
                (record_id, record_type, eval_revision),
            ).fetchall()
        if not rows:
            return None

        trial_spec = self._load_trial_spec(rows[0]["trial_hash"])
        stages = [self._load_timeline_stage_record(row) for row in rows]
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

    def get_timeline_view(
        self,
        record_id: str,
        record_type: Literal["trial", "candidate"],
        eval_revision: str,
    ) -> RecordTimelineView | None:
        """Return a rich timeline view with lineage, events, and observability."""
        if record_type == "trial":
            trial = self.get_trial_record(record_id, eval_revision)
            if trial is None or trial.trial_spec is None:
                return None
            timeline = self.get_record_timeline(record_id, "trial", eval_revision)
            if timeline is None:
                return None
            return RecordTimelineView(
                record_id=record_id,
                record_type="trial",
                trial_hash=trial.spec_hash,
                lineage=self._lineage_for_trial(trial.trial_spec, eval_revision),
                trial_spec=trial.trial_spec,
                item_payload=self._item_payload_for_trial(trial.spec_hash),
                timeline=timeline,
                observability=self._observability_for_view(
                    trial.spec_hash, None, eval_revision
                ),
                extractions=[],
                related_events=self._events_for_revision(
                    trial.spec_hash, eval_revision
                ),
            )

        trial_hash = self._lookup_trial_hash(record_id, eval_revision)
        if trial_hash is None:
            return None
        trial = self.get_trial_record(trial_hash, eval_revision)
        if trial is None or trial.trial_spec is None:
            return None
        timeline = self.get_record_timeline(record_id, "candidate", eval_revision)
        candidate = next(
            (
                candidate
                for candidate in trial.candidates
                if candidate.spec_hash == record_id
            ),
            None,
        )
        if timeline is None or candidate is None:
            return None
        return RecordTimelineView(
            record_id=record_id,
            record_type="candidate",
            trial_hash=trial_hash,
            candidate_id=record_id,
            lineage=self._lineage_for_trial(trial.trial_spec, eval_revision),
            trial_spec=trial.trial_spec,
            item_payload=self._item_payload_for_trial(trial_hash),
            timeline=timeline,
            conversation=candidate.conversation,
            inference=candidate.inference,
            extractions=candidate.extractions,
            evaluation=candidate.evaluation,
            judge_audit=self._load_judge_audit(candidate.judge_audits),
            observability=self._observability_for_view(
                trial_hash, record_id, eval_revision
            ),
            related_events=[
                event
                for event in self._events_for_revision(trial_hash, eval_revision)
                if event.candidate_id == record_id
            ],
        )

    def materialize_trial_record(
        self,
        trial_hash: str,
        eval_revision: str,
        *,
        extra_events: list[TrialEvent] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> TrialRecord:
        """Replay events into a trial record and refresh all projection tables."""
        if conn is None:
            with self.manager.get_connection() as local_conn:
                with local_conn:
                    return self.materialize_trial_record(
                        trial_hash,
                        eval_revision,
                        extra_events=extra_events,
                        conn=local_conn,
                    )

        trial_spec = self._load_trial_spec(trial_hash)
        events = self._events_for_revision(
            trial_hash,
            eval_revision,
            extra_events=extra_events,
        )
        if not events:
            raise ValueError(
                f"No events found for trial {trial_hash} and revision {eval_revision}."
            )

        candidate_states: dict[str, CandidateReplayState] = {}
        trial_stage_events: list[TrialEvent] = []
        candidate_stage_events: dict[str, list[TrialEvent]] = {}
        trial_status = RecordStatus.OK
        trial_error = None

        for event in events:
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
            candidate_id: self._build_timeline(
                record_id=candidate_id,
                record_type="candidate",
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
        trial_timeline = self._build_timeline(
            record_id=trial_hash,
            record_type="trial",
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

        self._upsert_trial_summary(conn, trial_record)
        self._delete_trial_metric_scores(conn, trial_hash, eval_revision)
        conn.execute(
            "DELETE FROM candidate_summary WHERE trial_hash = ? AND eval_revision = ?",
            (trial_hash, eval_revision),
        )
        for candidate in candidates:
            self._upsert_candidate_summary(
                conn,
                trial_hash=trial_hash,
                eval_revision=eval_revision,
                sample_index=candidate_states[candidate.spec_hash].sample_index,
                candidate=candidate,
            )
        self._replace_metric_scores(conn, candidates, eval_revision)
        conn.execute(
            "DELETE FROM record_timeline WHERE trial_hash = ? AND eval_revision = ?",
            (trial_hash, eval_revision),
        )
        self._insert_timeline(conn, trial_timeline, eval_revision)
        for candidate_timeline in candidate_timelines.values():
            self._insert_timeline(conn, candidate_timeline, eval_revision)

        return trial_record

    def iter_candidate_scores(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        eval_revision: str = "latest",
    ) -> Iterator[ScoreRow]:
        """Yield projected metric scores filtered by trial, metric, and revision."""
        clauses: list[str] = []
        params: list[object] = []
        if trial_hash is not None:
            clauses.append("candidate_summary.trial_hash = ?")
            params.append(trial_hash)
        if metric_id is not None:
            clauses.append("metric_scores.metric_id = ?")
            params.append(metric_id)
        clauses.append("candidate_summary.eval_revision = ?")
        params.append(eval_revision)
        clauses.append("metric_scores.eval_revision = candidate_summary.eval_revision")

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT candidate_summary.trial_hash, metric_scores.candidate_id, metric_scores.metric_id, metric_scores.score, metric_scores.details_json
            FROM metric_scores
            JOIN candidate_summary
              ON candidate_summary.candidate_id = metric_scores.candidate_id
             AND candidate_summary.eval_revision = metric_scores.eval_revision
            {where_clause}
            ORDER BY candidate_summary.trial_hash, metric_scores.candidate_id, metric_scores.metric_id
        """
        with self.manager.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        for row in rows:
            details = self._load_json_dict(
                row["details_json"],
                label="metric_scores.details_json",
                context=(
                    f"trial_hash={row['trial_hash']}, candidate_id={row['candidate_id']}, "
                    f"metric_id={row['metric_id']}"
                ),
                default={},
            )
            yield ScoreRow(
                trial_hash=row["trial_hash"],
                candidate_id=row["candidate_id"],
                metric_id=row["metric_id"],
                score=row["score"],
                details=details,
            )

    def iter_trial_summaries(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
    ) -> Iterator[TrialSummaryRow]:
        """Yield projected trial summary rows without hydrating full trial records."""
        if trial_hashes is not None and not trial_hashes:
            return

        query = """
            SELECT trial_hash, model_id, task_id, item_id, status
            FROM trial_summary
        """
        params: list[object] = []
        if trial_hashes:
            placeholders = ", ".join("?" for _ in trial_hashes)
            query += f" WHERE trial_hash IN ({placeholders})"
            params.extend(trial_hashes)
        query += " ORDER BY trial_hash ASC"

        with self.manager.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        for row in rows:
            yield TrialSummaryRow(
                trial_hash=row["trial_hash"],
                model_id=row["model_id"],
                task_id=row["task_id"],
                item_id=row["item_id"],
                status=row["status"],
            )

    def has_trial(self, trial_hash: str, eval_revision: str = "latest") -> bool:
        """Return whether a completed projected trial exists for the revision."""
        return (
            self.event_repo.latest_terminal_event_type(trial_hash)
            == TrialEventType.TRIAL_COMPLETED
            and self.event_repo.has_projection_for_revision(trial_hash, eval_revision)
        )

    def _load_trial_spec(self, trial_hash: str) -> TrialSpec:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                "SELECT canonical_json FROM specs WHERE spec_hash = ?",
                (trial_hash,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Missing persisted TrialSpec for {trial_hash}.")
        context = f"trial_hash={trial_hash}"
        payload = self._load_json_value(
            row["canonical_json"],
            label="specs.canonical_json",
            context=context,
            default=None,
        )
        try:
            return TrialSpec.model_validate(payload)
        except ValidationError as exc:
            raise self._storage_read_error(
                "specs.canonical_json", context, exc
            ) from exc

    def _lookup_trial_hash(self, candidate_id: str, eval_revision: str) -> str | None:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT trial_hash
                FROM candidate_summary
                WHERE candidate_id = ? AND eval_revision = ?
                """,
                (candidate_id, eval_revision),
            ).fetchone()
            if row is not None:
                return row["trial_hash"]
            row = conn.execute(
                "SELECT trial_hash FROM trial_events WHERE candidate_id = ? ORDER BY event_seq ASC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return row["trial_hash"] if row is not None else None

    def _has_materialized_trial(self, trial_hash: str, eval_revision: str) -> bool:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM candidate_summary
                WHERE trial_hash = ? AND eval_revision = ?
                LIMIT 1
                """,
                (trial_hash, eval_revision),
            ).fetchone()
        return row is not None

    def _events_for_revision(
        self,
        trial_hash: str,
        eval_revision: str,
        *,
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
            and event.metadata.get("eval_revision") == eval_revision
        ]
        if not projection_events:
            if extra_events is None and eval_revision == "latest":
                return events
            return []

        target_projection = projection_events[-1]
        return [
            event for event in events if event.event_seq <= target_projection.event_seq
        ]

    def _item_payload_for_trial(self, trial_hash: str) -> JSONDict | None:
        for event in self.event_repo.get_events(trial_hash):
            if event.stage != "item_load":
                continue
            if isinstance(event.payload, dict):
                return event.payload
            if self.artifact_store is not None:
                artifact_hash = next(
                    (
                        artifact.artifact_hash
                        for artifact in event.artifact_refs
                        if artifact.role == "item_payload"
                    ),
                    None,
                )
                if artifact_hash is None:
                    payload_hash = event.metadata.get("item_payload_hash")
                    artifact_hash = (
                        payload_hash if isinstance(payload_hash, str) else None
                    )
                if artifact_hash is not None:
                    payload = self.artifact_store.read_json(artifact_hash)
                    if isinstance(payload, dict):
                        return payload
        return None

    def _observability_for_view(
        self,
        trial_hash: str,
        candidate_id: str | None,
        eval_revision: str,
    ) -> ObservabilityRefs | None:
        if self.observability_store is None:
            return None
        refs = self.observability_store.get_refs(
            trial_hash, candidate_id, eval_revision
        )
        if refs is not None:
            return refs
        if candidate_id is None:
            return None
        return self.observability_store.get_refs(trial_hash, None, eval_revision)

    def _build_timeline(
        self,
        *,
        record_id: str,
        record_type: Literal["trial", "candidate"],
        trial_hash: str,
        candidate_id: str | None,
        item_id: str,
        events: list[TrialEvent],
    ) -> RecordTimeline:
        stages = [
            self._timeline_stage_from_event(event)
            for event in events
            if event.stage is not None
        ]
        source_range = (events[0].event_seq, events[-1].event_seq) if events else None
        return RecordTimeline(
            record_id=record_id,
            record_type=record_type,
            trial_hash=trial_hash,
            candidate_id=candidate_id,
            item_id=item_id,
            stages=stages,
            source_event_range=source_range,
        )

    def _timeline_stage_from_event(self, event: TrialEvent) -> TimelineStageRecord:
        stage = event.stage
        if stage is None:
            raise ValueError("Timeline stages require event.stage to be populated.")

        component_id: str | None = None
        if event.stage == "inference":
            provider = event.metadata.get("provider")
            model_id = event.metadata.get("model_id")
            if isinstance(provider, str) and isinstance(model_id, str):
                component_id = f"{provider}/{model_id}"
            elif isinstance(provider, str):
                component_id = provider
            elif isinstance(model_id, str):
                component_id = model_id
        elif event.stage == "extraction":
            extractor_id = event.metadata.get("extractor_id")
            component_id = extractor_id if isinstance(extractor_id, str) else None
        elif event.stage == "evaluation":
            metric_id = event.metadata.get("metric_id")
            component_id = metric_id if isinstance(metric_id, str) else None
        elif event.stage == "prompt_render":
            prompt_template_id = event.metadata.get("prompt_template_id")
            component_id = (
                prompt_template_id if isinstance(prompt_template_id, str) else None
            )
        elif event.stage == "projection":
            projection_version = event.metadata.get("projection_version")
            component_id = (
                projection_version if isinstance(projection_version, str) else None
            )

        return TimelineStageRecord(
            stage=stage,
            status=event.status
            or (RecordStatus.ERROR if event.error is not None else RecordStatus.OK),
            component_id=component_id,
            started_at=event.event_ts,
            ended_at=event.event_ts,
            duration_ms=0,
            metadata=event.metadata,
            artifact_refs=event.artifact_refs,
            error=event.error,
        )

    def _upsert_trial_summary(self, conn, record: TrialRecord) -> None:
        trial_spec = record.trial_spec
        timeline = record.timeline
        stages = timeline.stages if timeline is not None else []
        started_at = stages[0].started_at.isoformat() if stages else None
        ended_at = stages[-1].ended_at.isoformat() if stages else None
        duration_ms = sum(stage.duration_ms for stage in stages) if stages else None
        item_stage = next(
            (stage for stage in stages if stage.stage == "item_load"), None
        )
        tags = item_stage.metadata.get("tags") if item_stage is not None else None
        has_conversation = any(
            candidate.conversation is not None for candidate in record.candidates
        )
        has_logprobs = any(
            candidate.inference is not None and candidate.inference.logprobs is not None
            for candidate in record.candidates
        )
        has_trace = any(
            candidate.inference is not None
            and candidate.inference.reasoning_trace is not None
            for candidate in record.candidates
        )
        conn.execute(
            """
            INSERT INTO trial_summary (
                trial_hash,
                model_id,
                task_id,
                item_id,
                status,
                started_at,
                ended_at,
                duration_ms,
                has_conversation,
                has_logprobs,
                has_trace,
                tags_json,
                error_fingerprint,
                error_preview
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trial_hash) DO UPDATE SET
                model_id=excluded.model_id,
                task_id=excluded.task_id,
                item_id=excluded.item_id,
                status=excluded.status,
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                duration_ms=excluded.duration_ms,
                has_conversation=excluded.has_conversation,
                has_logprobs=excluded.has_logprobs,
                has_trace=excluded.has_trace,
                tags_json=excluded.tags_json,
                error_fingerprint=excluded.error_fingerprint,
                error_preview=excluded.error_preview,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                record.spec_hash,
                trial_spec.model.model_id if trial_spec else None,
                trial_spec.task.task_id if trial_spec else None,
                trial_spec.item_id if trial_spec else None,
                record.status.value,
                started_at,
                ended_at,
                duration_ms,
                int(has_conversation),
                int(has_logprobs),
                int(has_trace),
                json.dumps(tags) if isinstance(tags, dict) else None,
                record.error.fingerprint if record.error else None,
                record.error.message if record.error else None,
            ),
        )

    def _upsert_candidate_summary(
        self,
        conn,
        *,
        trial_hash: str,
        eval_revision: str,
        sample_index: int,
        candidate: CandidateRecord,
    ) -> None:
        inference = candidate.inference
        token_usage = (
            inference.token_usage if inference and inference.token_usage else None
        )
        conn.execute(
            """
            INSERT INTO candidate_summary (
                candidate_id,
                trial_hash,
                eval_revision,
                sample_index,
                status,
                finish_reason,
                tokens_in,
                tokens_out,
                latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id, eval_revision) DO UPDATE SET
                trial_hash=excluded.trial_hash,
                sample_index=excluded.sample_index,
                status=excluded.status,
                finish_reason=excluded.finish_reason,
                tokens_in=excluded.tokens_in,
                tokens_out=excluded.tokens_out,
                latency_ms=excluded.latency_ms
            """,
            (
                candidate.spec_hash,
                trial_hash,
                eval_revision,
                sample_index,
                candidate.status.value,
                None,
                token_usage.prompt_tokens if token_usage else None,
                token_usage.completion_tokens if token_usage else None,
                int(inference.latency_ms)
                if inference and inference.latency_ms is not None
                else None,
            ),
        )

    def _replace_metric_scores(
        self,
        conn,
        candidates: list[CandidateRecord],
        eval_revision: str,
    ) -> None:
        candidate_ids = [candidate.spec_hash for candidate in candidates]
        if candidate_ids:
            placeholders = ", ".join("?" for _ in candidate_ids)
            conn.execute(
                f"""
                DELETE FROM metric_scores
                WHERE eval_revision = ? AND candidate_id IN ({placeholders})
                """,
                [eval_revision, *candidate_ids],
            )

        for candidate in candidates:
            if candidate.evaluation is None:
                continue
            for score in candidate.evaluation.metric_scores:
                conn.execute(
                    """
                    INSERT INTO metric_scores (candidate_id, eval_revision, metric_id, score, details_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.spec_hash,
                        eval_revision,
                        score.metric_id,
                        score.value,
                        json.dumps(score.details) if score.details else None,
                    ),
                )

    def _delete_trial_metric_scores(
        self, conn, trial_hash: str, eval_revision: str
    ) -> None:
        conn.execute(
            """
            DELETE FROM metric_scores
            WHERE candidate_id IN (
                SELECT candidate_id
                FROM candidate_summary
                WHERE trial_hash = ? AND eval_revision = ?
            )
            AND eval_revision = ?
            """,
            (trial_hash, eval_revision, eval_revision),
        )

    def _insert_timeline(
        self, conn, timeline: RecordTimeline, eval_revision: str
    ) -> None:
        source_start = (
            timeline.source_event_range[0] if timeline.source_event_range else None
        )
        source_end = (
            timeline.source_event_range[1] if timeline.source_event_range else None
        )
        for stage_order, stage in enumerate(timeline.stages):
            conn.execute(
                """
                INSERT INTO record_timeline (
                    record_id,
                    record_type,
                    trial_hash,
                    eval_revision,
                    candidate_id,
                    stage_order,
                    stage_name,
                    status,
                    started_at,
                    ended_at,
                    duration_ms,
                    component_id,
                    metadata_json,
                    artifacts_json,
                    error_json,
                    source_start_seq,
                    source_end_seq
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timeline.record_id,
                    timeline.record_type,
                    timeline.trial_hash,
                    eval_revision,
                    timeline.candidate_id,
                    stage_order,
                    stage.stage,
                    stage.status.value,
                    stage.started_at.isoformat(),
                    stage.ended_at.isoformat(),
                    stage.duration_ms,
                    stage.component_id,
                    json.dumps(stage.metadata) if stage.metadata else None,
                    json.dumps(
                        [
                            artifact.model_dump(mode="json")
                            for artifact in stage.artifact_refs
                        ]
                    )
                    if stage.artifact_refs
                    else None,
                    json.dumps(stage.error.model_dump(mode="json"))
                    if stage.error
                    else None,
                    source_start,
                    source_end,
                ),
            )

    def _lineage_for_trial(
        self, trial_spec: TrialSpec, eval_revision: str
    ) -> dict[str, str | None]:
        return {
            "task": trial_spec.task.task_id,
            "model": trial_spec.model.model_id,
            "item": trial_spec.item_id,
            "prompt_template": trial_spec.prompt.id,
            "eval_revision": eval_revision,
        }

    def _load_judge_audit(self, artifact_hashes: list[str]) -> JudgeAuditTrail | None:
        if self.artifact_store is None or not artifact_hashes:
            return None

        trails: list[JudgeAuditTrail] = []
        for artifact_hash in artifact_hashes:
            try:
                payload = self.artifact_store.read_json(artifact_hash)
                trails.append(JudgeAuditTrail.model_validate(payload))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise self._storage_read_error(
                    "artifacts.judge_audit",
                    f"artifact_hash={artifact_hash}",
                    exc,
                ) from exc
        if not trails:
            return None
        if len(trails) == 1:
            return trails[0]

        judge_calls = []
        for trail in trails:
            judge_calls.extend(trail.judge_calls)
        return JudgeAuditTrail(
            spec_hash=trails[-1].spec_hash,
            candidate_hash=trails[-1].candidate_hash,
            judge_calls=judge_calls,
        )

    def _decode_json_column(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: object | None,
    ) -> object | None:
        if raw_value is None:
            return default
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise self._storage_read_error(label, context, exc) from exc

    def _load_json_dict(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: JSONDict,
    ) -> JSONDict:
        decoded = self._decode_json_column(
            raw_value,
            label=label,
            context=context,
            default=default,
        )
        try:
            return _JSON_DICT_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self._storage_read_error(label, context, exc) from exc

    def _load_json_list(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: JSONList,
    ) -> JSONList:
        decoded = self._decode_json_column(
            raw_value,
            label=label,
            context=context,
            default=default,
        )
        try:
            return _JSON_LIST_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self._storage_read_error(label, context, exc) from exc

    def _load_json_value(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: JSONValueType | None,
    ) -> JSONValueType | None:
        decoded = self._decode_json_column(
            raw_value,
            label=label,
            context=context,
            default=default,
        )
        if decoded is None:
            return None
        try:
            return _JSON_VALUE_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self._storage_read_error(label, context, exc) from exc

    def _load_timeline_stage_record(self, row: sqlite3.Row) -> TimelineStageRecord:
        context = (
            f"trial_hash={row['trial_hash']}, record_id={row['record_id']}, "
            f"stage={row['stage_name']}, eval_revision={row['eval_revision']}"
        )
        metadata = self._load_json_dict(
            row["metadata_json"],
            label="record_timeline.metadata_json",
            context=context,
            default={},
        )
        artifact_items = self._load_json_list(
            row["artifacts_json"],
            label="record_timeline.artifacts_json",
            context=context,
            default=[],
        )
        error_payload = self._decode_json_column(
            row["error_json"],
            label="record_timeline.error_json",
            context=context,
            default=None,
        )
        try:
            artifact_refs = [
                ArtifactRef.model_validate(item) for item in artifact_items
            ]
        except ValidationError as exc:
            raise self._storage_read_error(
                "record_timeline.artifacts_json", context, exc
            ) from exc
        try:
            error = (
                ErrorRecord.model_validate(error_payload)
                if error_payload is not None
                else None
            )
        except ValidationError as exc:
            raise self._storage_read_error(
                "record_timeline.error_json", context, exc
            ) from exc
        return TimelineStageRecord(
            stage=row["stage_name"],
            status=RecordStatus(row["status"]),
            component_id=row["component_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_ms=row["duration_ms"],
            metadata=metadata if isinstance(metadata, dict) else {},
            artifact_refs=artifact_refs,
            error=error,
        )

    def _storage_read_error(
        self,
        label: str,
        context: str,
        exc: ValidationError | json.JSONDecodeError,
    ) -> StorageError:
        detail = (
            format_validation_error(exc)
            if isinstance(exc, ValidationError)
            else exc.msg
        )
        return StorageError(
            code=ErrorCode.STORAGE_READ,
            message=f"Failed to hydrate {label} ({context}): {detail}",
        )
