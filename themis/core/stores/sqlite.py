"""SQLite-backed run store implementation."""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import sqlite3
from pathlib import Path

from themis.core.base import JSONValue
from themis.core.events import RunEvent, event_from_dict
from themis.core.snapshot import RunSnapshot, StoredRun, snapshot_from_dict
from themis.core.stores.base import ProjectionRefreshingStore


class SqliteRunStore(ProjectionRefreshingStore):
    """Small SQLite-backed run store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_snapshots (
                    run_id TEXT PRIMARY KEY,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_blobs (
                    blob_ref TEXT PRIMARY KEY,
                    media_type TEXT NOT NULL,
                    payload BLOB NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_projections (
                    run_id TEXT NOT NULL,
                    projection_name TEXT NOT NULL,
                    projection_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, projection_name)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_cache (
                    stage_name TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (stage_name, cache_key)
                )
                """
            )
            connection.commit()

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        snapshot_json = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_snapshots (run_id, snapshot_json)
                VALUES (?, ?)
                """,
                (snapshot.run_id, snapshot_json),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO run_projections (run_id, projection_name, projection_json)
                VALUES (?, ?, ?)
                """,
                (snapshot.run_id, "snapshot", snapshot_json),
            )
            connection.commit()
        self._bootstrap_projections(snapshot)

    def persist_event(self, event: RunEvent) -> None:
        event_json = json.dumps(event.model_dump(mode="json"), sort_keys=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                INSERT INTO run_events (run_id, event_type, event_json)
                VALUES (?, ?, ?)
                """,
                (event.run_id, event.event_type, event_json),
            )
            connection.commit()
        snapshot = self._load_snapshot(event.run_id)
        if snapshot is not None:
            self._refresh_projections_for_event(snapshot, event)

    def query_events(self, run_id: str) -> list[RunEvent]:
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                """
                SELECT event_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        events: list[RunEvent] = []
        for (payload,) in rows:
            event_payload = json.loads(payload)
            try:
                events.append(event_from_dict(event_payload))
            except KeyError:
                continue
        return events

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                """
                SELECT projection_json
                FROM run_projections
                WHERE run_id = ? AND projection_name = ?
                """,
                (run_id, projection_name),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def store_blob(self, blob: bytes, media_type: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        ref = f"sha256:{digest}"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO run_blobs (blob_ref, media_type, payload)
                VALUES (?, ?, ?)
                """,
                (ref, media_type, blob),
            )
            connection.commit()
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                """
                SELECT media_type, payload
                FROM run_blobs
                WHERE blob_ref = ?
                """,
                (blob_ref,),
            ).fetchone()
        if row is None:
            return None
        media_type, payload = row
        return media_type, bytes(payload)

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            return None
        return StoredRun(snapshot=snapshot, events=self.query_events(run_id))

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM stage_cache
                WHERE stage_name = ? AND cache_key = ?
                """,
                (stage_name, cache_key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO stage_cache (stage_name, cache_key, payload_json)
                VALUES (?, ?, ?)
                """,
                (stage_name, cache_key, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()

    def clear_run(self, run_id: str) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
            connection.execute(
                "DELETE FROM run_projections WHERE run_id = ?", (run_id,)
            )
            connection.execute("DELETE FROM run_snapshots WHERE run_id = ?", (run_id,))
            connection.commit()

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM run_snapshots
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return snapshot_from_dict(json.loads(row[0]))

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_projections (run_id, projection_name, projection_json)
                VALUES (?, ?, ?)
                """,
                (run_id, projection_name, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()


def sqlite_store(path: str | Path) -> SqliteRunStore:
    """Build a SQLite-backed store."""
    return SqliteRunStore(path)
