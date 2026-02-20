"""SQLite metadata storage backend."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from collections.abc import Iterator

from themis.storage.models import RunMetadata, RunStatus


class MetadataStore:
    """Manages SQLite metadata database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._write_lock = threading.RLock()
        self._init_database()

    def _init_database(self):
        """Initialize SQLite metadata database."""
        with self._write_lock:
            with self._connect() as conn:
                # WAL allows concurrent readers with a single writer and
                # significantly reduces lock contention in threaded CI runs.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        config TEXT,
                        tags TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        total_samples INTEGER DEFAULT 0,
                        successful_generations INTEGER DEFAULT 0,
                        failed_generations INTEGER DEFAULT 0,
                        config_snapshot TEXT,
                        error_message TEXT,
                        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        eval_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        eval_name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metrics_config TEXT,
                        total_evaluated INTEGER DEFAULT 0,
                        total_failures INTEGER DEFAULT 0,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_runs_experiment
                    ON runs(experiment_id)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_runs_status
                    ON runs(status)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_evaluations_run
                    ON evaluations(run_id)
                """)
                conn.commit()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Create a SQLite connection configured for concurrent access."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata to SQLite database."""
        # Serialise process-local writers to avoid lock thrash on Windows CI.
        with self._write_lock:
            retry_delay = 0.05
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    with self._connect() as conn:
                        # Ensure experiment exists
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO experiments (experiment_id, name, created_at, updated_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                metadata.experiment_id,
                                metadata.experiment_id,
                                metadata.created_at,
                                metadata.updated_at,
                            ),
                        )

                        # Upsert run
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO runs (
                                run_id, experiment_id, status, created_at, updated_at, completed_at,
                                total_samples, successful_generations, failed_generations,
                                config_snapshot, error_message
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                metadata.run_id,
                                metadata.experiment_id,
                                metadata.status.value,
                                metadata.created_at,
                                metadata.updated_at,
                                metadata.completed_at,
                                metadata.total_samples,
                                metadata.successful_generations,
                                metadata.failed_generations,
                                json.dumps(metadata.config_snapshot),
                                metadata.error_message,
                            ),
                        )
                        conn.commit()
                        return
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == max_attempts - 1:
                        raise
                    time.sleep(retry_delay)
                    retry_delay *= 2

    def delete_run(self, run_id: str) -> None:
        """Delete run from database."""
        with self._write_lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                conn.commit()

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        with self._connect() as conn:
            query = "SELECT * FROM runs WHERE 1=1"
            params = []

            if experiment_id:
                query += " AND experiment_id = ?"
                params.append(experiment_id)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        # Convert to RunMetadata
        runs = []
        for row in rows:
            runs.append(
                RunMetadata(
                    run_id=row[0],
                    experiment_id=row[1],
                    status=RunStatus(row[2]),
                    created_at=row[3],
                    updated_at=row[4],
                    completed_at=row[5],
                    total_samples=row[6] or 0,
                    successful_generations=row[7] or 0,
                    failed_generations=row[8] or 0,
                    config_snapshot=json.loads(row[9]) if row[9] else {},
                    error_message=row[10],
                )
            )

        return runs

    def get_run_experiment_id(self, run_id: str) -> str | None:
        """Resolve experiment ID for a run ID."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT experiment_id FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return row[0]
