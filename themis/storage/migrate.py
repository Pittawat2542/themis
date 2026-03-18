"""Utilities for copying persisted stores between supported backends."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
from typing import cast

from themis.errors import StorageError
from themis.records.observability import ObservabilityLink
from themis.specs.experiment import PostgresBlobStorageSpec
from themis.storage.blobs.local_fs import LocalBlobStore
from themis.storage._protocols import StorageConnectionManager
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.factory import StorageBundle, build_storage_bundle
from themis.types.enums import ErrorCode
from themis.types.events import ProjectionCompletedEventMetadata, TrialEventType


class _ReadOnlySqliteManager:
    """Minimal read-only connection manager for source-store migration reads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def migrate_sqlite_store(
    *,
    source_db_path: str | Path,
    destination_bundle: StorageBundle,
    source_blob_root: str | Path | None = None,
) -> None:
    """Copy a SQLite-backed store into another initialized storage bundle."""
    source_db = Path(source_db_path)
    source_manager = _ReadOnlySqliteManager(source_db)
    source_event_repo = SqliteEventRepository(
        cast(StorageConnectionManager, source_manager)
    )
    trial_hashes: list[str] = []

    with source_manager.get_connection() as src_conn:
        spec_rows = _load_table_rows(
            src_conn,
            "specs",
            columns={
                "spec_hash": None,
                "canonical_hash": "NULL",
                "spec_type": None,
                "schema_version": None,
                "canonical_json": None,
            },
            order_by="spec_hash ASC",
        )
        event_rows = src_conn.execute(
            """
            SELECT trial_hash, event_seq, event_id, candidate_id, event_type, stage, status,
                   event_ts, metadata_json, payload_json, artifact_refs_json, error_json
            FROM trial_events
            ORDER BY trial_hash ASC, event_seq ASC
            """
        ).fetchall()
        artifact_rows = src_conn.execute(
            """
            SELECT artifact_hash, path, size_bytes, compression, media_type
            FROM artifacts
            """
        ).fetchall()
        run_manifest_rows = _load_table_rows(
            src_conn,
            "run_manifests",
            columns={
                "run_id": None,
                "backend_kind": None,
                "project_spec_json": None,
                "benchmark_spec_json": "NULL",
                "experiment_spec_json": None,
                "manifest_json": None,
                "created_at": None,
            },
            order_by="run_id ASC",
        )
        stage_work_item_rows = _load_table_rows(
            src_conn,
            "stage_work_items",
            columns={
                "work_item_id": None,
                "run_id": None,
                "stage": None,
                "status": None,
                "trial_hash": None,
                "candidate_index": None,
                "candidate_id": None,
                "transform_hash": "NULL",
                "evaluation_hash": "NULL",
                "attempt_count": "0",
                "lease_owner": "NULL",
                "lease_expires_at": "NULL",
                "external_job_id": "NULL",
                "artifact_refs_json": "NULL",
                "started_at": "NULL",
                "ended_at": "NULL",
                "last_error_code": "NULL",
                "last_error_message": "NULL",
            },
            order_by="work_item_id ASC",
        )
        observability_rows = _load_observability_links(src_conn)
        trial_hashes = [
            row["spec_hash"] for row in spec_rows if row["spec_type"] == "TrialSpec"
        ]

    manager = destination_bundle.manager
    with manager.get_connection() as dst_conn:
        with dst_conn:
            for row in spec_rows:
                dst_conn.execute(
                    """
                    INSERT INTO specs (
                        spec_hash,
                        canonical_hash,
                        spec_type,
                        schema_version,
                        canonical_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(spec_hash) DO UPDATE SET
                        canonical_hash=excluded.canonical_hash,
                        spec_type=excluded.spec_type,
                        schema_version=excluded.schema_version,
                        canonical_json=excluded.canonical_json
                    """,
                    (
                        row["spec_hash"],
                        _canonical_hash_for_row(row),
                        row["spec_type"],
                        row["schema_version"],
                        row["canonical_json"],
                    ),
                )
            for row in event_rows:
                dst_conn.execute(
                    """
                    INSERT INTO trial_events (
                        trial_hash,
                        event_seq,
                        event_id,
                        candidate_id,
                        event_type,
                        stage,
                        status,
                        event_ts,
                        metadata_json,
                        payload_json,
                        artifact_refs_json,
                        error_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trial_hash, event_seq) DO UPDATE SET
                        event_id=excluded.event_id,
                        candidate_id=excluded.candidate_id,
                        event_type=excluded.event_type,
                        stage=excluded.stage,
                        status=excluded.status,
                        event_ts=excluded.event_ts,
                        metadata_json=excluded.metadata_json,
                        payload_json=excluded.payload_json,
                        artifact_refs_json=excluded.artifact_refs_json,
                        error_json=excluded.error_json
                    """,
                    (
                        row["trial_hash"],
                        row["event_seq"],
                        row["event_id"],
                        row["candidate_id"],
                        row["event_type"],
                        row["stage"],
                        row["status"],
                        row["event_ts"],
                        row["metadata_json"],
                        row["payload_json"],
                        row["artifact_refs_json"],
                        row["error_json"],
                    ),
                )
            for row in artifact_rows:
                dst_conn.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_hash,
                        path,
                        size_bytes,
                        compression,
                        media_type
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_hash) DO UPDATE SET
                        path=excluded.path,
                        size_bytes=excluded.size_bytes,
                        compression=excluded.compression,
                        media_type=excluded.media_type
                    """,
                    (
                        row["artifact_hash"],
                        row["path"],
                        row["size_bytes"],
                        row["compression"],
                        row["media_type"],
                    ),
                )
            for row in run_manifest_rows:
                dst_conn.execute(
                    """
                    INSERT INTO run_manifests (
                        run_id,
                        backend_kind,
                        project_spec_json,
                        benchmark_spec_json,
                        experiment_spec_json,
                        manifest_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        backend_kind=excluded.backend_kind,
                        project_spec_json=excluded.project_spec_json,
                        benchmark_spec_json=excluded.benchmark_spec_json,
                        experiment_spec_json=excluded.experiment_spec_json,
                        manifest_json=excluded.manifest_json,
                        created_at=excluded.created_at
                    """,
                    (
                        row["run_id"],
                        row["backend_kind"],
                        row["project_spec_json"],
                        row["benchmark_spec_json"],
                        row["experiment_spec_json"],
                        row["manifest_json"],
                        row["created_at"],
                    ),
                )
            for row in stage_work_item_rows:
                dst_conn.execute(
                    """
                    INSERT INTO stage_work_items (
                        work_item_id,
                        run_id,
                        stage,
                        status,
                        trial_hash,
                        candidate_index,
                        candidate_id,
                        transform_hash,
                        evaluation_hash,
                        attempt_count,
                        lease_owner,
                        lease_expires_at,
                        external_job_id,
                        artifact_refs_json,
                        started_at,
                        ended_at,
                        last_error_code,
                        last_error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(work_item_id) DO UPDATE SET
                        run_id=excluded.run_id,
                        stage=excluded.stage,
                        status=excluded.status,
                        trial_hash=excluded.trial_hash,
                        candidate_index=excluded.candidate_index,
                        candidate_id=excluded.candidate_id,
                        transform_hash=excluded.transform_hash,
                        evaluation_hash=excluded.evaluation_hash,
                        attempt_count=excluded.attempt_count,
                        lease_owner=excluded.lease_owner,
                        lease_expires_at=excluded.lease_expires_at,
                        external_job_id=excluded.external_job_id,
                        artifact_refs_json=excluded.artifact_refs_json,
                        started_at=excluded.started_at,
                        ended_at=excluded.ended_at,
                        last_error_code=excluded.last_error_code,
                        last_error_message=excluded.last_error_message
                    """,
                    (
                        row["work_item_id"],
                        row["run_id"],
                        row["stage"],
                        row["status"],
                        row["trial_hash"],
                        row["candidate_index"],
                        row["candidate_id"],
                        row["transform_hash"],
                        row["evaluation_hash"],
                        row["attempt_count"],
                        row["lease_owner"],
                        row["lease_expires_at"],
                        row["external_job_id"],
                        row["artifact_refs_json"],
                        row["started_at"],
                        row["ended_at"],
                        row["last_error_code"],
                        row["last_error_message"],
                    ),
                )

    for row in observability_rows:
        destination_bundle.observability_store.save_link(
            row["trial_hash"],
            row["candidate_id"] or None,
            row["overlay_key"],
            ObservabilityLink(
                provider=row["provider"],
                external_id=row["external_id"],
                url=row["url"],
                extras=json.loads(row["extras_json"] or "{}"),
            ),
        )

    if source_blob_root is not None and destination_bundle.blob_store is not None:
        destination_blob_store = cast(LocalBlobStore, destination_bundle.blob_store)
        _copy_blob_tree(
            Path(source_blob_root),
            Path(destination_blob_store.base_path),
        )

    for trial_hash in trial_hashes:
        destination_bundle.projection_repo.materialize_trial_record(trial_hash)
        overlays = {
            (
                event.metadata.transform_hash,
                event.metadata.evaluation_hash,
            )
            for event in source_event_repo.get_events(trial_hash)
            if event.event_type == TrialEventType.PROJECTION_COMPLETED
            and isinstance(event.metadata, ProjectionCompletedEventMetadata)
        }
        for transform_hash, evaluation_hash in overlays:
            destination_bundle.projection_repo.materialize_trial_record(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )


def migrate_sqlite_to_postgres(
    *,
    source_db_path: str | Path,
    database_url: str,
    blob_root_dir: str | Path,
    source_blob_root: str | Path | None = None,
) -> StorageBundle:
    """Copy a SQLite-backed store into a Postgres-backed storage bundle."""
    destination_bundle = build_storage_bundle(
        PostgresBlobStorageSpec(
            database_url=database_url,
            blob_root_dir=str(blob_root_dir),
        )
    )
    migrate_sqlite_store(
        source_db_path=source_db_path,
        destination_bundle=destination_bundle,
        source_blob_root=source_blob_root,
    )
    return destination_bundle


def _sqlite_table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _sqlite_table_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _load_table_rows(
    conn,
    table_name: str,
    *,
    columns: dict[str, str | None],
    order_by: str | None = None,
) -> list:
    if not _sqlite_table_exists(conn, table_name):
        return []
    existing_columns = _sqlite_table_columns(conn, table_name)
    select_list: list[str] = []
    for column_name, missing_expr in columns.items():
        if column_name in existing_columns:
            select_list.append(column_name)
            continue
        if missing_expr is None:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=(
                    f"unsupported source store format: expected column "
                    f"'{column_name}' in table '{table_name}'."
                ),
            )
        select_list.append(f"{missing_expr} AS {column_name}")
    query = f"SELECT {', '.join(select_list)} FROM {table_name}"
    if order_by is not None:
        query += f" ORDER BY {order_by}"
    return conn.execute(query).fetchall()


def _canonical_hash_for_row(row) -> str:
    canonical_hash = row["canonical_hash"]
    if canonical_hash:
        return str(canonical_hash)
    canonical_json = row["canonical_json"]
    return hashlib.sha256(str(canonical_json).encode("utf-8")).hexdigest()


def _load_observability_links(conn) -> list:
    if not _sqlite_table_exists(conn, "observability_links"):
        if _sqlite_table_exists(conn, "observability_refs"):
            legacy_ref_count = conn.execute(
                "SELECT COUNT(*) AS ref_count FROM observability_refs"
            ).fetchone()["ref_count"]
            if legacy_ref_count:
                raise StorageError(
                    code=ErrorCode.STORAGE_READ,
                    message=(
                        "unsupported source store format: expected observability_links "
                        "rows instead of legacy observability_refs"
                    ),
                )
        return []
    return conn.execute(
        """
        SELECT trial_hash, candidate_id, overlay_key, provider, external_id, url, extras_json
        FROM observability_links
        ORDER BY trial_hash ASC, candidate_id ASC, overlay_key ASC, provider ASC
        """
    ).fetchall()


def _copy_blob_tree(source_root: Path, destination_root: Path) -> None:
    if not source_root.exists():
        return
    shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
