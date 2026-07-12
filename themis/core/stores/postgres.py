"""Postgres-backed run store with filesystem blob storage."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from themis.core.base import JSONValue
from themis.core.events import RunEvent, event_from_dict
from themis.core.registry import RunRecord
from themis.core.results import ExecutionCheckpoint, ProjectionCursor
from themis.core.snapshot import RunSnapshot, StoredRun, snapshot_from_dict
from themis.core.store import AppendResult, EventRecord
from themis.core.stores.base import ProjectionRefreshingStore

_SCHEMA_VERSION = "2"


class PostgresRunStore(ProjectionRefreshingStore):
    """Persist runs in Postgres while storing blobs on the filesystem."""

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
            if current_version == "1":
                raise RuntimeError(
                    "Unsupported schema-v1 Themis store; archive or reset it before using v5."
                )
            if current_version == "0":
                self._migrate_to_v1(connection)
            self._ensure_checkpoint_schema(connection)
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

    def persist_event(self, event: RunEvent) -> AppendResult:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT sequence FROM run_events WHERE run_id = %s AND event_id = %s",
                (event.run_id, event.event_id),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return AppendResult(sequence=int(existing["sequence"]), inserted=False)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence FROM run_events WHERE run_id = %s",
                (event.run_id,),
            ).fetchone()
            sequence = int(row["sequence"])
            connection.execute(
                """
                INSERT INTO run_events (run_id, event_id, sequence, event_type, event_json)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    event.run_id,
                    event.event_id,
                    sequence,
                    event.event_type,
                    json.dumps(event.model_dump(mode="json"), sort_keys=True),
                ),
            )
            connection.commit()
        return AppendResult(sequence=sequence, inserted=True)

    def query_events(self, run_id: str) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_json::text AS event_json
                FROM run_events
                WHERE run_id = %s
                ORDER BY sequence ASC
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

    def query_event_records(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[EventRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_json::text AS event_json
                FROM run_events
                WHERE run_id = %s AND sequence > %s
                ORDER BY sequence ASC LIMIT %s
                """,
                (run_id, after_sequence, limit),
            ).fetchall()
        return [
            EventRecord(
                sequence=int(row["sequence"]),
                event=event_from_dict(json.loads(row["event_json"])),
            )
            for row in rows
        ]

    def count_events(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS event_count
                FROM run_events
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
        return int(row["event_count"]) if row is not None else 0

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

    def load_execution_checkpoint(self, run_id: str) -> ExecutionCheckpoint | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_json::text AS checkpoint_json
                FROM execution_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ExecutionCheckpoint.model_validate_json(row["checkpoint_json"])

    def store_execution_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO execution_checkpoints (run_id, checkpoint_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE
                SET checkpoint_json = EXCLUDED.checkpoint_json
                """,
                (checkpoint.run_id, checkpoint.model_dump_json()),
            )
            connection.commit()

    def load_projection_cursor(
        self, run_id: str, projection_name: str
    ) -> ProjectionCursor | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT cursor_json::text AS cursor_json
                FROM projection_cursors
                WHERE run_id = %s AND projection_name = %s
                """,
                (run_id, projection_name),
            ).fetchone()
        if row is None:
            return None
        return ProjectionCursor.model_validate_json(row["cursor_json"])

    def store_projection_cursor(self, cursor: ProjectionCursor) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projection_cursors (run_id, projection_name, cursor_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (run_id, projection_name) DO UPDATE
                SET cursor_json = EXCLUDED.cursor_json
                """,
                (cursor.run_id, cursor.projection_name, cursor.model_dump_json()),
            )
            connection.commit()

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
        return StoredRun(
            snapshot=snapshot,
            events=self.query_events(run_id),
            execution_checkpoint=self.load_execution_checkpoint(run_id),
            event_count=self.count_events(run_id),
        )

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

    def _write_run_record(self, run_id: str, record: RunRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_registry (run_id, record_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE
                SET record_json = EXCLUDED.record_json
                """,
                (run_id, record.model_dump_json()),
            )
            connection.commit()

    def _read_run_record(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_json::text AS record_json
                FROM run_registry
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunRecord.model_validate_json(row["record_json"])

    def _list_run_records(self) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_json::text AS record_json
                FROM run_registry
                ORDER BY run_id ASC
                """
            ).fetchall()
        return [RunRecord.model_validate_json(row["record_json"]) for row in rows]

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
                "DELETE FROM execution_checkpoints WHERE run_id = %s", (run_id,)
            )
            connection.execute(
                "DELETE FROM projection_cursors WHERE run_id = %s", (run_id,)
            )
            connection.execute(
                "DELETE FROM run_projections WHERE run_id = %s", (run_id,)
            )
            connection.execute("DELETE FROM run_snapshots WHERE run_id = %s", (run_id,))
            connection.execute("DELETE FROM run_registry WHERE run_id = %s", (run_id,))
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
                event_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                event_type TEXT NOT NULL,
                event_json JSONB NOT NULL,
                UNIQUE (run_id, event_id),
                UNIQUE (run_id, sequence)
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
            CREATE TABLE IF NOT EXISTS run_registry (
                run_id TEXT PRIMARY KEY,
                record_json JSONB NOT NULL
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
        self._ensure_checkpoint_schema(connection)

    def _ensure_checkpoint_schema(self, connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_checkpoints (
                run_id TEXT PRIMARY KEY,
                checkpoint_json JSONB NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS projection_cursors (
                run_id TEXT NOT NULL,
                projection_name TEXT NOT NULL,
                cursor_json JSONB NOT NULL,
                PRIMARY KEY (run_id, projection_name)
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
    """Create a Postgres-backed run store."""

    return PostgresRunStore(url, blob_root)
