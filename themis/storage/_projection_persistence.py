"""Internal SQL write helpers for materialized projection tables."""

from __future__ import annotations

import json

from themis.records.candidate import CandidateRecord
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.storage._protocols import StorageConnection
from themis.types.events import TimelineStage


class ProjectionWriter:
    """Persists projected trial summaries, candidate rows, metrics, and timelines."""

    def upsert_trial_summary(
        self,
        conn: StorageConnection,
        record: TrialRecord,
        overlay_key: str,
    ) -> None:
        trial_spec = record.trial_spec
        timeline = record.timeline
        stages = timeline.stages if timeline is not None else []
        started_at = stages[0].started_at.isoformat() if stages else None
        ended_at = stages[-1].ended_at.isoformat() if stages else None
        duration_ms = sum(stage.duration_ms for stage in stages) if stages else None
        item_stage = next(
            (stage for stage in stages if stage.stage == TimelineStage.ITEM_LOAD), None
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
                overlay_key,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trial_hash, overlay_key) DO UPDATE SET
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
                overlay_key,
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

    def upsert_candidate_summary(
        self,
        conn: StorageConnection,
        *,
        trial_hash: str,
        overlay_key: str,
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
                overlay_key,
                sample_index,
                status,
                finish_reason,
                tokens_in,
                tokens_out,
                latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id, overlay_key) DO UPDATE SET
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
                overlay_key,
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

    def replace_metric_scores(
        self,
        conn: StorageConnection,
        candidates: list[CandidateRecord],
        overlay_key: str,
    ) -> None:
        candidate_ids = [candidate.spec_hash for candidate in candidates]
        if candidate_ids:
            placeholders = ", ".join("?" for _ in candidate_ids)
            conn.execute(
                f"""
                DELETE FROM metric_scores
                WHERE overlay_key = ? AND candidate_id IN ({placeholders})
                """,
                [overlay_key, *candidate_ids],
            )

        for candidate in candidates:
            if candidate.evaluation is None:
                continue
            for score in candidate.evaluation.metric_scores:
                conn.execute(
                    """
                    INSERT INTO metric_scores (candidate_id, overlay_key, metric_id, score, details_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.spec_hash,
                        overlay_key,
                        score.metric_id,
                        score.value,
                        json.dumps(score.details) if score.details else None,
                    ),
                )

    def delete_trial_metric_scores(
        self,
        conn: StorageConnection,
        trial_hash: str,
        overlay_key: str,
    ) -> None:
        conn.execute(
            """
            DELETE FROM metric_scores
            WHERE candidate_id IN (
                SELECT candidate_id
                FROM candidate_summary
                WHERE trial_hash = ? AND overlay_key = ?
            )
            AND overlay_key = ?
            """,
            (trial_hash, overlay_key, overlay_key),
        )

    def insert_timeline(
        self,
        conn: StorageConnection,
        timeline: RecordTimeline,
        overlay_key: str,
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
                    overlay_key,
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
                    timeline.record_type.value,
                    timeline.trial_hash,
                    overlay_key,
                    timeline.candidate_id,
                    stage_order,
                    stage.stage.value,
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
