"""Postgres connection management behind the shared storage contract."""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from collections.abc import Sequence
from typing import Iterator, Protocol

from themis._optional import import_optional
from themis.errors import StorageError
from themis.storage._protocols import StorageConnection, StorageCursor
from themis.storage._schema import (
    SCHEMA,
    STORE_FORMAT_KEY,
    STORE_FORMAT_VERSION,
    apply_sql_script,
)
from themis.types.enums import ErrorCode

_QMARK_PATTERN = re.compile(r"\?")


class _RawPostgresConnection(Protocol):
    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> StorageCursor: ...

    def __enter__(self) -> object: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...


class _PostgresStorageConnection:
    """Minimal Postgres adapter matching the storage contract."""

    def __init__(self, connection: _RawPostgresConnection) -> None:
        self._connection = connection

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> StorageCursor:
        translated = _QMARK_PATTERN.sub("%s", query)
        return self._connection.execute(translated, params or ())

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._connection.__exit__(exc_type, exc, tb)


class PostgresConnectionManager:
    """Manages Postgres connections and schema initialization."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._local = threading.local()

    @contextmanager
    def get_connection(self) -> Iterator[StorageConnection]:
        """Yields a live storage connection, creating one per thread as needed."""

        if self._needs_new_connection():
            psycopg = import_optional("psycopg", extra="storage-postgres")
            rows = getattr(psycopg, "rows")
            self._local.conn = psycopg.connect(
                self.database_url,
                row_factory=rows.dict_row,
            )
        yield _PostgresStorageConnection(self._local.conn)

    def _needs_new_connection(self) -> bool:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return True
        return bool(getattr(conn, "closed", False) or getattr(conn, "broken", False))

    def initialize(self) -> None:
        """Creates the storage schema and validates the store format version."""

        with self.get_connection() as conn:
            with conn:
                apply_sql_script(conn, SCHEMA)
                self._migrate(conn)
                self._ensure_store_format(conn)

    def _ensure_store_format(self, conn: StorageConnection) -> None:
        row = conn.execute(
            """
            SELECT metadata_value
            FROM store_metadata
            WHERE metadata_key = ?
            """,
            (STORE_FORMAT_KEY,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO store_metadata (metadata_key, metadata_value)
                VALUES (?, ?)
                """,
                (STORE_FORMAT_KEY, STORE_FORMAT_VERSION),
            )
            return
        if row["metadata_value"] == "stage_overlays_v2":
            conn.execute(
                """
                UPDATE store_metadata
                SET metadata_value = ?
                WHERE metadata_key = ?
                """,
                (STORE_FORMAT_VERSION, STORE_FORMAT_KEY),
            )
            return
        if row["metadata_value"] != STORE_FORMAT_VERSION:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=(
                    "unsupported store format: expected "
                    f"{STORE_FORMAT_VERSION}, found {row['metadata_value']}"
                ),
                details={"database_url": self.database_url},
            )

    def _migrate(self, conn: StorageConnection) -> None:
        for statement in (
            "ALTER TABLE specs ADD COLUMN IF NOT EXISTS canonical_hash TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS benchmark_id TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS slice_id TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS prompt_variant_id TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS dimensions_json TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS started_at TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS ended_at TEXT",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS duration_ms INTEGER",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS has_conversation INTEGER DEFAULT 0",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS has_logprobs INTEGER DEFAULT 0",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS has_trace INTEGER DEFAULT 0",
            "ALTER TABLE trial_summary ADD COLUMN IF NOT EXISTS tags_json TEXT",
            "ALTER TABLE stage_work_items ADD COLUMN IF NOT EXISTS started_at TEXT",
            "ALTER TABLE stage_work_items ADD COLUMN IF NOT EXISTS ended_at TEXT",
            "ALTER TABLE stage_work_items ADD COLUMN IF NOT EXISTS last_error_code TEXT",
            "ALTER TABLE stage_work_items ADD COLUMN IF NOT EXISTS last_error_message TEXT",
            "ALTER TABLE run_manifests ADD COLUMN IF NOT EXISTS benchmark_spec_json TEXT",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_specs_canonical_hash ON specs(canonical_hash)",
        ):
            conn.execute(statement)
