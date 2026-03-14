"""Internal structural protocols for storage connections and managers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ContextManager, Protocol, runtime_checkable


class StorageRow(Protocol):
    """Minimal row surface used by storage readers."""

    def __getitem__(self, key: str) -> Any: ...


class StorageCursor(Protocol):
    """Minimal DB-API cursor surface used by storage repositories."""

    def fetchone(self) -> StorageRow | None: ...

    def fetchall(self) -> list[StorageRow]: ...


@runtime_checkable
class StorageConnection(Protocol):
    """Shared SQL-connection surface used by SQLite and Postgres backends."""

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> StorageCursor: ...

    def __enter__(self) -> StorageConnection: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...


@runtime_checkable
class StorageConnectionManager(Protocol):
    """Minimal connection-manager surface required by storage collaborators."""

    def get_connection(self) -> ContextManager[StorageConnection]: ...

    def initialize(self) -> None: ...
