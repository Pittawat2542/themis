"""Public storage constructors for Themis evidence."""

from __future__ import annotations

from themis.core.config import StorageConfig
from themis.core.store import (
    AppendResult,
    EventRecord,
    ProjectionConsistency,
    ProjectionFreshness,
    ProjectionRead,
    RunStore,
)
from themis.core.stores.jsonl import JsonlRunStore, jsonl_store
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.mongodb import MongoDbRunStore, mongodb_store
from themis.core.stores.postgres import PostgresRunStore, postgres_store
from themis.core.stores.sqlite import SqliteRunStore, sqlite_store
from themis.core.stores.factory import (
    available_store_backends,
    create_run_store,
    register_store_backend,
)
from themis.core.stores.base import RunStoreBase


def memory_store() -> InMemoryRunStore:
    """Create an ephemeral evidence store."""

    return InMemoryRunStore()


def storage_config(store: RunStore) -> StorageConfig:
    """Describe concrete storage as execution provenance."""

    if isinstance(store, InMemoryRunStore):
        return StorageConfig(target="memory")
    if isinstance(store, SqliteRunStore):
        return StorageConfig(target="sqlite", kwargs={"path": str(store.path)})
    if isinstance(store, JsonlRunStore):
        return StorageConfig(target="jsonl", kwargs={"root": str(store.root)})
    if isinstance(store, PostgresRunStore):
        return StorageConfig(
            target="postgres",
            kwargs={"url": store.url, "blob_root": str(store.blob_root)},
        )
    if isinstance(store, MongoDbRunStore):
        return StorageConfig(
            target="mongodb",
            kwargs={
                "url": store.url,
                "database": store.database,
                "blob_root": str(store.blob_root),
            },
        )
    declared = getattr(store, "storage_config", None)
    if isinstance(declared, StorageConfig):
        return declared
    raise TypeError("Custom stores must expose a StorageConfig through storage_config.")


__all__ = [
    "AppendResult",
    "EventRecord",
    "InMemoryRunStore",
    "JsonlRunStore",
    "MongoDbRunStore",
    "PostgresRunStore",
    "ProjectionConsistency",
    "ProjectionFreshness",
    "ProjectionRead",
    "RunStore",
    "RunStoreBase",
    "SqliteRunStore",
    "StorageConfig",
    "available_store_backends",
    "create_run_store",
    "jsonl_store",
    "memory_store",
    "mongodb_store",
    "postgres_store",
    "register_store_backend",
    "sqlite_store",
]
