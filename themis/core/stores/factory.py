"""Backend registry and store factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from themis.core.config import StorageConfig
from themis.core.store import RunStore
from themis.core.stores.jsonl import jsonl_store
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.sqlite import sqlite_store

StoreBuilder = Callable[[StorageConfig], RunStore]


def memory_store() -> InMemoryRunStore:
    return InMemoryRunStore()


_STORE_BUILDERS: dict[str, StoreBuilder] = {
    "memory": lambda config: memory_store(),
    "jsonl": lambda config: jsonl_store(Path(str(config.parameters.get("root", "runs/jsonl")))),
    "sqlite": lambda config: sqlite_store(Path(str(config.parameters.get("path", "runs/themis.sqlite3")))),
}


def register_store_backend(name: str, builder: StoreBuilder) -> None:
    _STORE_BUILDERS[name] = builder


def available_store_backends() -> list[str]:
    return sorted(_STORE_BUILDERS)


def create_run_store(config: StorageConfig) -> RunStore:
    try:
        builder = _STORE_BUILDERS[config.store]
    except KeyError as exc:
        raise ValueError(f"Unsupported store backend: {config.store}") from exc
    return builder(config)
