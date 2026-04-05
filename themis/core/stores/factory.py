"""Backend registry and store factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from themis.core.config import StorageConfig
from themis.core.store import RunStore
from themis.core.stores.jsonl import jsonl_store
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.mongodb import mongodb_store
from themis.core.stores.postgres import postgres_store
from themis.core.stores.sqlite import sqlite_store

StoreBuilder = Callable[[StorageConfig], RunStore]


def memory_store() -> InMemoryRunStore:
    """Create an in-memory run store."""

    return InMemoryRunStore()


_STORE_BUILDERS: dict[str, StoreBuilder] = {
    "memory": lambda config: memory_store(),
    "jsonl": lambda config: jsonl_store(
        Path(str(config.parameters.get("root", "runs/jsonl")))
    ),
    "mongodb": lambda config: mongodb_store(
        str(config.parameters.get("url", "mongodb://localhost:27017")),
        str(config.parameters.get("database", "themis")),
        Path(str(config.parameters.get("blob_root", "runs/mongodb-blobs"))),
    ),
    "postgres": lambda config: postgres_store(
        str(config.parameters.get("url", "postgresql://localhost/themis")),
        Path(str(config.parameters.get("blob_root", "runs/postgres-blobs"))),
    ),
    "sqlite": lambda config: sqlite_store(
        Path(str(config.parameters.get("path", "runs/themis.sqlite3")))
    ),
}


def register_store_backend(name: str, builder: StoreBuilder) -> None:
    """Register a custom run-store backend for `create_run_store`."""

    _STORE_BUILDERS[name] = builder


def available_store_backends() -> list[str]:
    """Return the registered storage backend names."""

    return sorted(_STORE_BUILDERS)


def create_run_store(config: StorageConfig) -> RunStore:
    """Instantiate a run store from storage configuration."""

    try:
        builder = _STORE_BUILDERS[config.store]
    except KeyError as exc:
        raise ValueError(f"Unsupported store backend: {config.store}") from exc
    return builder(config)
