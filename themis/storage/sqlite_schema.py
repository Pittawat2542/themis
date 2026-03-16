"""SQLite schema and connection management for the local storage backend."""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from themis.errors import StorageError
from themis.storage._schema import (
    SCHEMA,
    STORE_FORMAT_KEY,
    STORE_FORMAT_VERSION,
    THEMIS_TABLES,
    apply_sql_script,
)
from themis.types.enums import ErrorCode


class DatabaseManager:
    """Manages SQLite connections and schema initialization."""

    def __init__(self, uri: str):
        if not uri.startswith("sqlite:///"):
            raise ValueError("URI must standard 'sqlite:///' format.")

        self.db_path = uri.replace("sqlite:///", "")
        self._local = threading.local()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Provides a thread-local SQLite connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None,
            )
            self._local.conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn.execute("PRAGMA journal_mode = WAL;")
            self._local.conn.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn.row_factory = sqlite3.Row

        try:
            yield self._local.conn
        except Exception:
            raise

    def initialize(self) -> None:
        """Create tables, indexes, and lightweight additive migrations."""
        with self.get_connection() as conn:
            with conn:
                self._reject_unsupported_store_format(conn)
                apply_sql_script(conn, SCHEMA)
                self._ensure_store_format(conn)
                self._migrate(conn)

    def _reject_unsupported_store_format(self, conn: sqlite3.Connection) -> None:
        existing_tables = self._existing_user_tables(conn)
        if not existing_tables:
            return
        if "store_metadata" in existing_tables:
            return
        if existing_tables & THEMIS_TABLES:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=(
                    "unsupported store format: expected store_metadata row "
                    f"{STORE_FORMAT_KEY}={STORE_FORMAT_VERSION}"
                ),
                details={"db_path": self.db_path},
            )

    def _ensure_store_format(self, conn: sqlite3.Connection) -> None:
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
        if row["metadata_value"] != STORE_FORMAT_VERSION:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=(
                    "unsupported store format: expected "
                    f"{STORE_FORMAT_VERSION}, found {row['metadata_value']}"
                ),
                details={"db_path": self.db_path},
            )

    def _existing_user_tables(self, conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        return {row["name"] for row in rows}

    def _migrate(self, conn: sqlite3.Connection) -> None:
        self._ensure_columns(
            conn,
            "specs",
            {
                "canonical_hash": "TEXT",
            },
        )
        self._ensure_columns(
            conn,
            "trial_summary",
            {
                "started_at": "TEXT",
                "ended_at": "TEXT",
                "duration_ms": "INTEGER",
                "has_conversation": "INTEGER DEFAULT 0",
                "has_logprobs": "INTEGER DEFAULT 0",
                "has_trace": "INTEGER DEFAULT 0",
                "tags_json": "TEXT",
            },
        )
        self._ensure_columns(
            conn,
            "stage_work_items",
            {
                "started_at": "TEXT",
                "ended_at": "TEXT",
                "last_error_code": "TEXT",
                "last_error_message": "TEXT",
            },
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_specs_canonical_hash ON specs(canonical_hash)"
        )

    def _ensure_columns(
        self,
        conn: sqlite3.Connection,
        table: str,
        columns: dict[str, str],
    ) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
