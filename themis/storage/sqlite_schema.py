"""SQLite schema and connection management for the local storage backend."""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS specs (
    spec_hash TEXT PRIMARY KEY,
    spec_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    compression TEXT,
    media_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trial_summary (
    trial_hash TEXT PRIMARY KEY,
    model_id TEXT,
    task_id TEXT,
    item_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    duration_ms INTEGER,
    has_conversation INTEGER DEFAULT 0,
    has_logprobs INTEGER DEFAULT 0,
    has_trace INTEGER DEFAULT 0,
    tags_json TEXT,
    error_fingerprint TEXT,
    error_preview TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trial_events (
    trial_hash TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    candidate_id TEXT,
    event_type TEXT NOT NULL,
    stage TEXT,
    status TEXT,
    event_ts TEXT NOT NULL,
    metadata_json TEXT,
    payload_json TEXT,
    artifact_refs_json TEXT,
    error_json TEXT,
    PRIMARY KEY (trial_hash, event_seq),
    FOREIGN KEY(trial_hash) REFERENCES specs(spec_hash)
);

CREATE TABLE IF NOT EXISTS candidate_summary (
    candidate_id TEXT NOT NULL,
    trial_hash TEXT NOT NULL,
    eval_revision TEXT NOT NULL DEFAULT 'latest',
    sample_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    finish_reason TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    PRIMARY KEY (candidate_id, eval_revision),
    FOREIGN KEY(trial_hash) REFERENCES specs(spec_hash)
);

CREATE TABLE IF NOT EXISTS metric_scores (
    candidate_id TEXT NOT NULL,
    eval_revision TEXT NOT NULL DEFAULT 'latest',
    metric_id TEXT NOT NULL,
    score REAL NOT NULL,
    details_json TEXT,
    FOREIGN KEY(candidate_id, eval_revision) REFERENCES candidate_summary(candidate_id, eval_revision),
    PRIMARY KEY (candidate_id, metric_id, eval_revision)
);

CREATE TABLE IF NOT EXISTS record_timeline (
    record_id TEXT NOT NULL,
    record_type TEXT NOT NULL,
    trial_hash TEXT NOT NULL,
    eval_revision TEXT NOT NULL DEFAULT 'latest',
    candidate_id TEXT,
    stage_order INTEGER NOT NULL,
    stage_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    component_id TEXT,
    metadata_json TEXT,
    artifacts_json TEXT,
    error_json TEXT,
    source_start_seq INTEGER,
    source_end_seq INTEGER,
    PRIMARY KEY (record_id, stage_order, eval_revision),
    FOREIGN KEY(trial_hash) REFERENCES specs(spec_hash)
);

CREATE TABLE IF NOT EXISTS observability_refs (
    trial_hash TEXT NOT NULL,
    candidate_id TEXT NOT NULL DEFAULT '',
    eval_revision TEXT NOT NULL DEFAULT 'latest',
    langfuse_trace_id TEXT,
    langfuse_url TEXT,
    wandb_url TEXT,
    extras_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trial_hash, candidate_id, eval_revision)
);

CREATE INDEX IF NOT EXISTS idx_trial_events_trial_seq ON trial_events(trial_hash, event_seq);
CREATE INDEX IF NOT EXISTS idx_trial_events_candidate_seq ON trial_events(trial_hash, candidate_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_candidate_trial_hash ON candidate_summary(trial_hash, eval_revision);
CREATE INDEX IF NOT EXISTS idx_metric_candidate_hash ON metric_scores(candidate_id, eval_revision);
CREATE INDEX IF NOT EXISTS idx_record_timeline_trial ON record_timeline(trial_hash, eval_revision, record_type, candidate_id);
CREATE INDEX IF NOT EXISTS idx_observability_trial ON observability_refs(trial_hash, eval_revision, candidate_id);
"""


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
                conn.executescript(SCHEMA)
                self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
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
