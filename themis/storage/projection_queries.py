"""Internal SQL read/query helpers for SQLite projection reads."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from pydantic import TypeAdapter, ValidationError

from themis.records.conversation import Conversation, ConversationEvent
from themis.specs.experiment import TrialSpec
from themis.storage._projection_codec import ProjectionCodecs
from themis.storage._protocols import StorageConnectionManager
from themis.storage.event_repo import SqliteEventRepository
from themis.types.events import ScoreRow, TrialEventType, TrialSummaryRow
from themis.overlays import overlay_key_for

_CONVERSATION_EVENT_ADAPTER: TypeAdapter[ConversationEvent] = TypeAdapter(
    ConversationEvent
)


def _coerce_dimensions(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


class ProjectionQueries:
    """Owns SQL-backed read queries and basic projection hydration helpers."""

    def __init__(
        self,
        manager: StorageConnectionManager,
        event_repo: SqliteEventRepository,
        codecs: ProjectionCodecs,
    ) -> None:
        self.manager = manager
        self.event_repo = event_repo
        self.codecs = codecs

    def overlay_key(
        self,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> str:
        return overlay_key_for(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def overlay_exists_or_materialized(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        overlay_key: str | None = None,
    ) -> bool:
        resolved_overlay_key = overlay_key or self.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        return bool(
            self.event_repo.has_projection_for_overlay(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )
            or self.has_materialized_trial(trial_hash, resolved_overlay_key)
        )

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
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

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[ScoreRow]:
        if trial_hashes is not None and not trial_hashes:
            return

        overlay_key = self.overlay_key(evaluation_hash=evaluation_hash)
        clauses: list[str] = []
        params: list[object] = []
        if trial_hashes:
            placeholders = ", ".join("?" for _ in trial_hashes)
            clauses.append(f"candidate_summary.trial_hash IN ({placeholders})")
            params.extend(trial_hashes)
        if metric_id is not None:
            clauses.append("metric_scores.metric_id = ?")
            params.append(metric_id)
        clauses.append("candidate_summary.overlay_key = ?")
        params.append(overlay_key)
        clauses.append("metric_scores.overlay_key = candidate_summary.overlay_key")

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT candidate_summary.trial_hash, metric_scores.candidate_id, metric_scores.metric_id, metric_scores.score, metric_scores.details_json
            FROM metric_scores
            JOIN candidate_summary
              ON candidate_summary.candidate_id = metric_scores.candidate_id
             AND candidate_summary.overlay_key = metric_scores.overlay_key
            {where_clause}
            ORDER BY candidate_summary.trial_hash, metric_scores.candidate_id, metric_scores.metric_id
        """
        with self.manager.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        for row in rows:
            details = self.codecs.load_json_dict(
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
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[TrialSummaryRow]:
        if trial_hashes is not None and not trial_hashes:
            return

        overlay_key = self.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        query = """
            SELECT trial_hash, benchmark_id, model_id, task_id, slice_id, prompt_variant_id, dimensions_json, item_id, status
            FROM trial_summary
            WHERE overlay_key = ?
        """
        params: list[object] = [overlay_key]
        if trial_hashes:
            placeholders = ", ".join("?" for _ in trial_hashes)
            query += f" AND trial_hash IN ({placeholders})"
            params.extend(trial_hashes)
        query += " ORDER BY trial_hash ASC"

        with self.manager.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        for row in rows:
            yield TrialSummaryRow(
                trial_hash=row["trial_hash"],
                benchmark_id=row["benchmark_id"],
                model_id=row["model_id"],
                task_id=row["task_id"],
                slice_id=row["slice_id"],
                prompt_variant_id=row["prompt_variant_id"],
                dimensions=_coerce_dimensions(
                    self.codecs.load_json_value(
                        row["dimensions_json"],
                        label="trial_summary.dimensions_json",
                        context=f"trial_hash={row['trial_hash']}",
                        default={},
                    )
                ),
                item_id=row["item_id"],
                status=row["status"],
            )

    def has_trial(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        overlay_key = self.overlay_key(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        return self.event_repo.has_projection_for_overlay(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        ) and self.has_successful_trial(trial_hash, overlay_key)

    def load_trial_spec(self, trial_hash: str) -> TrialSpec:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                "SELECT canonical_json FROM specs WHERE spec_hash = ?",
                (trial_hash,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Missing persisted TrialSpec for {trial_hash}.")
        context = f"trial_hash={trial_hash}"
        payload = self.codecs.load_json_value(
            row["canonical_json"],
            label="specs.canonical_json",
            context=context,
            default=None,
        )
        try:
            return TrialSpec.model_validate(payload)
        except ValidationError as exc:
            raise self.codecs.storage_read_error(
                "specs.canonical_json", context, exc
            ) from exc

    def lookup_trial_hash(self, candidate_id: str, overlay_key: str) -> str | None:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT trial_hash
                FROM candidate_summary
                WHERE candidate_id = ? AND overlay_key = ?
                """,
                (candidate_id, overlay_key),
            ).fetchone()
            if row is not None:
                return row["trial_hash"]
            row = conn.execute(
                "SELECT trial_hash FROM trial_events WHERE candidate_id = ? ORDER BY event_seq ASC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return row["trial_hash"] if row is not None else None

    def has_materialized_trial(self, trial_hash: str, overlay_key: str) -> bool:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM trial_summary
                WHERE trial_hash = ? AND overlay_key = ?
                LIMIT 1
                """,
                (trial_hash, overlay_key),
            ).fetchone()
        return row is not None

    def has_successful_trial(self, trial_hash: str, overlay_key: str) -> bool:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT status
                FROM trial_summary
                WHERE trial_hash = ? AND overlay_key = ?
                LIMIT 1
                """,
                (trial_hash, overlay_key),
            ).fetchone()
        return row is not None and row["status"] == "ok"
