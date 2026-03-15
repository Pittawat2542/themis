"""Utilities for copying persisted stores between supported backends."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import cast

from themis.errors import StorageError
from themis.records.observability import ObservabilityLink
from themis.specs.experiment import PostgresBlobStorageSpec
from themis.storage.blobs.local_fs import LocalBlobStore
from themis.storage._protocols import StorageConnectionManager
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.factory import StorageBundle, build_storage_bundle
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import ErrorCode
from themis.types.events import ProjectionCompletedEventMetadata, TrialEventType


def migrate_sqlite_store(
    *,
    source_db_path: str | Path,
    destination_bundle: StorageBundle,
    source_blob_root: str | Path | None = None,
) -> None:
    """Copy a SQLite-backed store into another initialized storage bundle."""
    source_db = Path(source_db_path)
    source_manager = DatabaseManager(f"sqlite:///{source_db}")
    source_event_repo = SqliteEventRepository(
        cast(StorageConnectionManager, source_manager)
    )
    trial_hashes: list[str] = []

    with source_manager.get_connection() as src_conn:
        spec_rows = src_conn.execute(
            """
            SELECT spec_hash, canonical_hash, spec_type, schema_version, canonical_json
            FROM specs
            """
        ).fetchall()
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
        run_manifest_rows = _load_optional_rows(
            src_conn,
            "run_manifests",
            """
            SELECT run_id, backend_kind, project_spec_json, experiment_spec_json,
                   manifest_json, created_at
            FROM run_manifests
            """,
        )
        stage_work_item_rows = _load_optional_rows(
            src_conn,
            "stage_work_items",
            """
            SELECT work_item_id, run_id, stage, status, trial_hash, candidate_index,
                   candidate_id, transform_hash, evaluation_hash, attempt_count,
                   lease_owner, lease_expires_at, external_job_id, artifact_refs_json
            FROM stage_work_items
            """,
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
                        row["canonical_hash"],
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
                        experiment_spec_json,
                        manifest_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        backend_kind=excluded.backend_kind,
                        project_spec_json=excluded.project_spec_json,
                        experiment_spec_json=excluded.experiment_spec_json,
                        manifest_json=excluded.manifest_json,
                        created_at=excluded.created_at
                    """,
                    (
                        row["run_id"],
                        row["backend_kind"],
                        row["project_spec_json"],
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
                        artifact_refs_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        artifact_refs_json=excluded.artifact_refs_json
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


def _load_optional_rows(conn, table_name: str, query: str) -> list:
    if not _sqlite_table_exists(conn, table_name):
        return []
    return conn.execute(query).fetchall()


def _load_observability_links(conn) -> list:
    if not _sqlite_table_exists(conn, "observability_links"):
        if _sqlite_table_exists(conn, "observability_refs"):
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=(
                    "unsupported source store format: expected observability_links "
                    "table for migration."
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
