"""Postgres-backed run store with filesystem blob storage."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from themis.core.base import JSONValue
from themis.core.events import RunEvent, event_from_dict
from themis.core.snapshot import RunSnapshot, StoredRun, snapshot_from_dict
from themis.core.stores.base import ProjectionRefreshingStore


class PostgresRunStore(ProjectionRefreshingStore):
    def __init__(self, url: str, blob_root: str | Path) -> None:
        self.url = url
        self.blob_root = Path(blob_root)

    def initialize(self) -> None:
        self.blob_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
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
                CREATE TABLE IF NOT EXISTS run_projections (
                    run_id TEXT NOT NULL,
                    projection_name TEXT NOT NULL,
                    projection_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, projection_name)
                )
                """
            )
            connection.commit()

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_snapshots (run_id, snapshot_json)
                VALUES (%s, %s)
                """,
                (snapshot.run_id, json.dumps(snapshot.model_dump(mode="json"), sort_keys=True)),
            )
            connection.commit()
        self._refresh_projections(snapshot.run_id)

    def persist_event(self, event: RunEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_events (run_id, event_type, event_json)
                VALUES (%s, %s, %s)
                """,
                (event.run_id, event.event_type, json.dumps(event.model_dump(mode="json"), sort_keys=True)),
            )
            connection.commit()
        self._refresh_projections(event.run_id)

    def query_events(self, run_id: str) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_json
                FROM run_events
                WHERE run_id = %s
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        events: list[RunEvent] = []
        for row in rows:
            try:
                events.append(event_from_dict(json.loads(row["event_json"])))
            except KeyError:
                continue
        return events

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT projection_json
                FROM run_projections
                WHERE run_id = %s AND projection_name = %s
                """,
                (run_id, projection_name),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["projection_json"])

    def store_blob(self, blob: bytes, media_type: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        ref = f"sha256:{digest}"
        blob_path = self.blob_root / f"{digest}.blob"
        meta_path = self.blob_root / f"{digest}.meta.json"
        if not blob_path.exists():
            blob_path.write_bytes(blob)
        if not meta_path.exists():
            meta_path.write_text(json.dumps({"media_type": media_type}, sort_keys=True), encoding="utf-8")
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        digest = blob_ref.removeprefix("sha256:")
        blob_path = self.blob_root / f"{digest}.blob"
        meta_path = self.blob_root / f"{digest}.meta.json"
        if not blob_path.is_file() or not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))["media_type"], blob_path.read_bytes()

    def resume(self, run_id: str) -> StoredRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM run_snapshots
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredRun(
            snapshot=snapshot_from_dict(json.loads(row["snapshot_json"])),
            events=self.query_events(run_id),
        )

    def _write_projection(self, run_id: str, projection_name: str, payload: JSONValue) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_projections (run_id, projection_name, projection_json)
                VALUES (%s, %s, %s)
                """,
                (run_id, projection_name, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()

    def _connect(self):
        try:
            psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise ImportError("Postgres support requires the optional 'postgres' dependency.") from exc
        rows = getattr(psycopg, "rows", None)
        row_factory = getattr(rows, "dict_row", None) if rows is not None else None
        return psycopg.connect(self.url, row_factory=row_factory)


def postgres_store(url: str, blob_root: str | Path) -> PostgresRunStore:
    return PostgresRunStore(url, blob_root)
