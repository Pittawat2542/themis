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

_SCHEMA_VERSION = "1"


class PostgresRunStore(ProjectionRefreshingStore):
    def __init__(self, url: str, blob_root: str | Path) -> None:
        self.url = url
        self.blob_root = Path(blob_root)

    def initialize(self) -> None:
        self.blob_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                """
                SELECT value
                FROM run_store_meta
                WHERE key = %s
                """,
                ("schema_version",),
            ).fetchone()
            current_version = str(row["value"]) if row is not None else "0"
            if current_version == "0":
                self._migrate_to_v1(connection)
            connection.execute(
                """
                INSERT INTO run_store_meta (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                ("schema_version", _SCHEMA_VERSION),
            )
            connection.commit()

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_snapshots (run_id, snapshot_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE
                SET snapshot_json = EXCLUDED.snapshot_json
                """,
                (
                    snapshot.run_id,
                    json.dumps(snapshot.model_dump(mode="json"), sort_keys=True),
                ),
            )
            connection.commit()
        self._bootstrap_projections(snapshot)

    def persist_event(self, event: RunEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_events (run_id, event_type, event_json)
                VALUES (%s, %s, %s::jsonb)
                """,
                (
                    event.run_id,
                    event.event_type,
                    json.dumps(event.model_dump(mode="json"), sort_keys=True),
                ),
            )
            connection.commit()
        snapshot = self._load_snapshot(event.run_id)
        if snapshot is not None:
            self._refresh_projections_for_event(snapshot, event)

    def query_events(self, run_id: str) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_json::text AS event_json
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
        return self._get_projection_with_backfill(run_id, projection_name)

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT projection_json::text AS projection_json
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
            meta_path.write_text(
                json.dumps({"media_type": media_type}, sort_keys=True), encoding="utf-8"
            )
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        digest = blob_ref.removeprefix("sha256:")
        blob_path = self.blob_root / f"{digest}.blob"
        meta_path = self.blob_root / f"{digest}.meta.json"
        if not blob_path.is_file() or not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))[
            "media_type"
        ], blob_path.read_bytes()

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            return None
        return StoredRun(snapshot=snapshot, events=self.query_events(run_id))

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT snapshot_json::text AS snapshot_json
                FROM run_snapshots
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return snapshot_from_dict(json.loads(row["snapshot_json"]))

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_projections (run_id, projection_name, projection_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (run_id, projection_name) DO UPDATE
                SET projection_json = EXCLUDED.projection_json
                """,
                (run_id, projection_name, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json::text AS payload_json
                FROM stage_cache
                WHERE stage_name = %s AND cache_key = %s
                """,
                (stage_name, cache_key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO stage_cache (stage_name, cache_key, payload_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (stage_name, cache_key) DO UPDATE
                SET payload_json = EXCLUDED.payload_json
                """,
                (stage_name, cache_key, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()

    def clear_run(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM run_events WHERE run_id = %s", (run_id,))
            connection.execute(
                "DELETE FROM run_projections WHERE run_id = %s", (run_id,)
            )
            connection.execute("DELETE FROM run_snapshots WHERE run_id = %s", (run_id,))
            connection.commit()

    def _migrate_to_v1(self, connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_snapshots (
                run_id TEXT PRIMARY KEY,
                snapshot_json JSONB NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_events (
                id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_json JSONB NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_run_events_run_id_id
            ON run_events (run_id, id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_projections (
                run_id TEXT NOT NULL,
                projection_name TEXT NOT NULL,
                projection_json JSONB NOT NULL,
                PRIMARY KEY (run_id, projection_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_cache (
                stage_name TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                PRIMARY KEY (stage_name, cache_key)
            )
            """
        )

    def _connect(self):
        try:
            psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise ImportError(
                "Postgres support requires the optional 'postgres' dependency."
            ) from exc
        rows = getattr(psycopg, "rows", None)
        row_factory = getattr(rows, "dict_row", None) if rows is not None else None
        return psycopg.connect(self.url, row_factory=row_factory)


def postgres_store(url: str, blob_root: str | Path) -> PostgresRunStore:
    return PostgresRunStore(url, blob_root)
