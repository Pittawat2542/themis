from __future__ import annotations

import json

from themis.records.observability import (
    ObservabilityLink,
    ObservabilityRefs,
    ObservabilitySnapshot,
)
from themis.storage._protocols import StorageConnectionManager


class SqliteObservabilityStore:
    """Projection-side persistence for non-domain observability links."""

    def __init__(self, manager: StorageConnectionManager):
        self.manager = manager

    def save_snapshot(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        snapshot: ObservabilitySnapshot,
    ) -> None:
        with self.manager.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    DELETE FROM observability_links
                    WHERE trial_hash = ? AND candidate_id = ? AND overlay_key = ?
                    """,
                    (trial_hash, candidate_id or "", overlay_key),
                )
                for link in snapshot.links:
                    self.save_link(
                        trial_hash,
                        candidate_id,
                        overlay_key,
                        link,
                    )

    def save_link(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        link: ObservabilityLink,
    ) -> None:
        with self.manager.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO observability_links (
                        trial_hash,
                        candidate_id,
                        overlay_key,
                        provider,
                        external_id,
                        url,
                        extras_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trial_hash, candidate_id, overlay_key, provider) DO UPDATE SET
                        external_id=excluded.external_id,
                        url=excluded.url,
                        extras_json=excluded.extras_json,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        trial_hash,
                        candidate_id or "",
                        overlay_key,
                        link.provider,
                        link.external_id,
                        link.url,
                        json.dumps(link.extras, sort_keys=True),
                    ),
                )

    def get_snapshot(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
    ) -> ObservabilitySnapshot | None:
        with self.manager.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT provider, external_id, url, extras_json
                FROM observability_links
                WHERE trial_hash = ? AND candidate_id = ? AND overlay_key = ?
                ORDER BY provider ASC
                """,
                (trial_hash, candidate_id or "", overlay_key),
            ).fetchall()
        if not rows:
            return None
        return ObservabilitySnapshot(
            links=[
                ObservabilityLink(
                    provider=row["provider"],
                    external_id=row["external_id"],
                    url=row["url"],
                    extras=json.loads(row["extras_json"] or "{}"),
                )
                for row in rows
            ]
        )

    def save_refs(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        refs: ObservabilityRefs,
    ) -> None:
        self.save_snapshot(
            trial_hash,
            candidate_id,
            overlay_key,
            ObservabilitySnapshot(links=list(refs.links)),
        )

    def get_refs(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
    ) -> ObservabilityRefs | None:
        snapshot = self.get_snapshot(trial_hash, candidate_id, overlay_key)
        if snapshot is None:
            return None
        return ObservabilityRefs(links=list(snapshot.links))
