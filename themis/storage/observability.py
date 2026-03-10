from __future__ import annotations

import json

from themis.records.observability import ObservabilityRefs
from themis.storage.sqlite_schema import DatabaseManager


class SqliteObservabilityStore:
    """Projection-side persistence for non-domain observability links."""

    def __init__(self, manager: DatabaseManager):
        self.manager = manager

    def save_refs(
        self,
        trial_hash: str,
        candidate_id: str | None,
        eval_revision: str,
        refs: ObservabilityRefs,
    ) -> None:
        with self.manager.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO observability_refs (
                        trial_hash,
                        candidate_id,
                        eval_revision,
                        langfuse_trace_id,
                        langfuse_url,
                        wandb_url,
                        extras_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trial_hash, candidate_id, eval_revision) DO UPDATE SET
                        langfuse_trace_id=excluded.langfuse_trace_id,
                        langfuse_url=excluded.langfuse_url,
                        wandb_url=excluded.wandb_url,
                        extras_json=excluded.extras_json,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        trial_hash,
                        candidate_id or "",
                        eval_revision,
                        refs.langfuse_trace_id,
                        refs.langfuse_url,
                        refs.wandb_url,
                        json.dumps(refs.extras, sort_keys=True),
                    ),
                )

    def get_refs(
        self,
        trial_hash: str,
        candidate_id: str | None,
        eval_revision: str,
    ) -> ObservabilityRefs | None:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT langfuse_trace_id, langfuse_url, wandb_url, extras_json
                FROM observability_refs
                WHERE trial_hash = ? AND candidate_id = ? AND eval_revision = ?
                """,
                (trial_hash, candidate_id or "", eval_revision),
            ).fetchone()
        if row is None:
            return None
        return ObservabilityRefs(
            langfuse_trace_id=row["langfuse_trace_id"],
            langfuse_url=row["langfuse_url"],
            wandb_url=row["wandb_url"],
            extras=json.loads(row["extras_json"] or "{}"),
        )
