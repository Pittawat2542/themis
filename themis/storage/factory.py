"""Storage bundle construction for supported persistence backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from themis._optional import import_optional
from themis.contracts.protocols import (
    BlobStore,
    ObservabilityStore,
    ProjectionRepository,
    TrialEventRepository,
)
from themis.specs.experiment import PostgresBlobStorageSpec, SqliteBlobStorageSpec
from themis.storage.blobs.local_fs import LocalBlobStore
from themis.storage._protocols import StorageConnectionManager
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.postgres import PostgresConnectionManager
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import CompressionCodec


@dataclass(frozen=True, slots=True)
class StorageBundle:
    """Concrete storage collaborators for one configured backend."""

    backend: str
    event_repo: TrialEventRepository
    projection_repo: ProjectionRepository
    blob_store: BlobStore | None
    observability_store: ObservabilityStore
    manager: StorageConnectionManager


def build_storage_bundle(
    storage: SqliteBlobStorageSpec | PostgresBlobStorageSpec,
) -> StorageBundle:
    """Build the storage collaborators for one project-level storage config."""
    if isinstance(storage, SqliteBlobStorageSpec):
        root_dir = Path(storage.root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        sqlite_manager = DatabaseManager(f"sqlite:///{root_dir / 'themis.sqlite3'}")
        sqlite_manager.initialize()
        sqlite_storage_manager = cast(StorageConnectionManager, sqlite_manager)
        sqlite_blob_store: LocalBlobStore | None = (
            LocalBlobStore(root_dir / "artifacts", manager=sqlite_storage_manager)
            if storage.compression == CompressionCodec.ZSTD
            else None
        )
        observability_store = SqliteObservabilityStore(sqlite_storage_manager)
        return StorageBundle(
            backend=storage.backend.value,
            event_repo=SqliteEventRepository(sqlite_storage_manager),
            projection_repo=SqliteProjectionRepository(
                sqlite_storage_manager,
                artifact_store=sqlite_blob_store,
                observability_store=observability_store,
            ),
            blob_store=sqlite_blob_store,
            observability_store=observability_store,
            manager=sqlite_storage_manager,
        )

    import_optional("psycopg", extra="storage-postgres")
    blob_root = Path(storage.blob_root_dir)
    blob_root.mkdir(parents=True, exist_ok=True)
    postgres_manager = PostgresConnectionManager(storage.database_url)
    postgres_manager.initialize()
    postgres_storage_manager = cast(StorageConnectionManager, postgres_manager)
    postgres_blob_store: LocalBlobStore | None = (
        LocalBlobStore(blob_root, manager=postgres_storage_manager)
        if storage.compression == CompressionCodec.ZSTD
        else None
    )
    observability_store = SqliteObservabilityStore(postgres_storage_manager)
    return StorageBundle(
        backend=storage.backend.value,
        event_repo=SqliteEventRepository(postgres_storage_manager),
        projection_repo=SqliteProjectionRepository(
            postgres_storage_manager,
            artifact_store=postgres_blob_store,
            observability_store=observability_store,
        ),
        blob_store=postgres_blob_store,
        observability_store=observability_store,
        manager=postgres_storage_manager,
    )
