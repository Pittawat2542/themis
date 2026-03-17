"""Shared logical schema contract for supported storage backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class StatementExecutor(Protocol):
    """Minimal execution surface needed for schema application."""

    def execute(self, query: str, /) -> object: ...


STORE_FORMAT_KEY = "store_format"
STORE_FORMAT_VERSION = "stage_overlays_v3"

THEMIS_TABLES = {
    "specs",
    "artifacts",
    "trial_summary",
    "trial_events",
    "candidate_summary",
    "metric_scores",
    "record_timeline",
    "observability_refs",
    "observability_links",
    "run_manifests",
    "stage_work_items",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS specs (
    spec_hash TEXT PRIMARY KEY,
    canonical_hash TEXT NOT NULL,
    spec_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS store_metadata (
    metadata_key TEXT PRIMARY KEY,
    metadata_value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    trial_hash TEXT NOT NULL,
    overlay_key TEXT NOT NULL DEFAULT 'gen',
    benchmark_id TEXT,
    model_id TEXT,
    task_id TEXT,
    slice_id TEXT,
    prompt_variant_id TEXT,
    dimensions_json TEXT,
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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trial_hash, overlay_key)
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
    overlay_key TEXT NOT NULL DEFAULT 'gen',
    sample_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    finish_reason TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    PRIMARY KEY (candidate_id, overlay_key),
    FOREIGN KEY(trial_hash) REFERENCES specs(spec_hash)
);

CREATE TABLE IF NOT EXISTS metric_scores (
    candidate_id TEXT NOT NULL,
    overlay_key TEXT NOT NULL DEFAULT 'gen',
    metric_id TEXT NOT NULL,
    score REAL NOT NULL,
    details_json TEXT,
    FOREIGN KEY(candidate_id, overlay_key) REFERENCES candidate_summary(candidate_id, overlay_key),
    PRIMARY KEY (candidate_id, metric_id, overlay_key)
);

CREATE TABLE IF NOT EXISTS record_timeline (
    record_id TEXT NOT NULL,
    record_type TEXT NOT NULL,
    trial_hash TEXT NOT NULL,
    overlay_key TEXT NOT NULL DEFAULT 'gen',
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
    PRIMARY KEY (record_id, stage_order, overlay_key),
    FOREIGN KEY(trial_hash) REFERENCES specs(spec_hash)
);

CREATE TABLE IF NOT EXISTS observability_refs (
    trial_hash TEXT NOT NULL,
    candidate_id TEXT NOT NULL DEFAULT '',
    overlay_key TEXT NOT NULL DEFAULT 'gen',
    langfuse_trace_id TEXT,
    langfuse_url TEXT,
    wandb_url TEXT,
    extras_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trial_hash, candidate_id, overlay_key)
);

CREATE TABLE IF NOT EXISTS observability_links (
    trial_hash TEXT NOT NULL,
    candidate_id TEXT NOT NULL DEFAULT '',
    overlay_key TEXT NOT NULL DEFAULT 'gen',
    provider TEXT NOT NULL,
    external_id TEXT,
    url TEXT,
    extras_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trial_hash, candidate_id, overlay_key, provider)
);

CREATE TABLE IF NOT EXISTS run_manifests (
    run_id TEXT PRIMARY KEY,
    backend_kind TEXT NOT NULL,
    project_spec_json TEXT,
    benchmark_spec_json TEXT,
    experiment_spec_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_work_items (
    work_item_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    trial_hash TEXT NOT NULL,
    candidate_index INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    transform_hash TEXT,
    evaluation_hash TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at TEXT,
    external_job_id TEXT,
    artifact_refs_json TEXT,
    started_at TEXT,
    ended_at TEXT,
    last_error_code TEXT,
    last_error_message TEXT,
    FOREIGN KEY(run_id) REFERENCES run_manifests(run_id)
);

CREATE INDEX IF NOT EXISTS idx_trial_events_trial_seq ON trial_events(trial_hash, event_seq);
CREATE INDEX IF NOT EXISTS idx_trial_events_candidate_seq ON trial_events(trial_hash, candidate_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_trial_summary_overlay ON trial_summary(trial_hash, overlay_key);
CREATE INDEX IF NOT EXISTS idx_candidate_trial_hash ON candidate_summary(trial_hash, overlay_key);
CREATE INDEX IF NOT EXISTS idx_metric_candidate_hash ON metric_scores(candidate_id, overlay_key);
CREATE INDEX IF NOT EXISTS idx_record_timeline_trial ON record_timeline(trial_hash, overlay_key, record_type, candidate_id);
CREATE INDEX IF NOT EXISTS idx_observability_trial ON observability_refs(trial_hash, overlay_key, candidate_id);
CREATE INDEX IF NOT EXISTS idx_observability_links_trial ON observability_links(trial_hash, overlay_key, candidate_id, provider);
CREATE INDEX IF NOT EXISTS idx_stage_work_items_run_stage ON stage_work_items(run_id, stage, status);
"""


def iter_sql_statements(script: str) -> Iterator[str]:
    """Split a SQL script into individual executable statements."""
    for statement in script.split(";"):
        cleaned = statement.strip()
        if cleaned:
            yield cleaned


def apply_sql_script(conn: StatementExecutor, script: str) -> None:
    """Apply a SQL script using plain execute calls."""
    for statement in iter_sql_statements(script):
        conn.execute(statement)
