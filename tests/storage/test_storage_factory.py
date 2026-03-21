from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from themis.errors import ThemisError
from themis.specs.experiment import PostgresBlobStorageSpec, SqliteBlobStorageSpec
from themis.storage._protocols import StorageConnection, StorageConnectionManager
from themis.storage.artifact_store import ArtifactStore
from themis.storage import LocalBlobStore, build_storage_bundle
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.postgres import PostgresConnectionManager
from themis.storage.projection_materializer import ProjectionMaterializer
from themis.storage.projection_queries import ProjectionQueries
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.timeline_views import ProjectionTimelineViews
from themis.storage.factory import StorageBundle
from themis.types.enums import ErrorCode


def test_build_storage_bundle_supports_sqlite_blob(tmp_path: Path) -> None:
    bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "runs"))
    )

    assert isinstance(bundle.event_repo, SqliteEventRepository)
    assert isinstance(bundle.projection_repo, SqliteProjectionRepository)
    assert isinstance(bundle.observability_store, SqliteObservabilityStore)
    assert isinstance(bundle.blob_store, LocalBlobStore)


def test_build_storage_bundle_requires_postgres_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from themis import storage as module

    def raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(module.factory, "import_optional", raise_missing_optional)

    with pytest.raises(ThemisError, match=r"themis-eval\[storage-postgres\]"):
        build_storage_bundle(
            PostgresBlobStorageSpec(
                database_url="postgresql://localhost:5432/themis",
                blob_root_dir=str(tmp_path / "blobs"),
            )
        )


def test_storage_stack_uses_connection_manager_protocol(tmp_path):
    assert get_type_hints(StorageBundle)["manager"] is StorageConnectionManager
    assert (
        get_type_hints(ArtifactStore.__init__)["manager"]
        == StorageConnectionManager | None
    )
    assert (
        get_type_hints(SqliteEventRepository.__init__)["manager"]
        is StorageConnectionManager
    )
    assert (
        get_type_hints(SqliteObservabilityStore.__init__)["manager"]
        is StorageConnectionManager
    )
    assert (
        get_type_hints(SqliteProjectionRepository.__init__)["manager"]
        is StorageConnectionManager
    )
    assert (
        get_type_hints(ProjectionQueries.__init__)["manager"]
        is StorageConnectionManager
    )
    assert (
        get_type_hints(ProjectionMaterializer.__init__)["manager"]
        is StorageConnectionManager
    )
    assert (
        get_type_hints(ProjectionTimelineViews.__init__)["manager"]
        is StorageConnectionManager
    )

    bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "protocol-runs"))
    )

    assert isinstance(bundle.manager, StorageConnectionManager)
    assert isinstance(
        PostgresConnectionManager("postgresql://localhost:5432/themis"),
        StorageConnectionManager,
    )


def test_postgres_bundle_uses_shared_storage_repository_classes(tmp_path, monkeypatch):
    from themis import storage as module

    monkeypatch.setattr(
        module.factory, "import_optional", lambda module_name, *, extra: object()
    )
    monkeypatch.setattr(PostgresConnectionManager, "initialize", lambda self: None)

    bundle = build_storage_bundle(
        PostgresBlobStorageSpec(
            database_url="postgresql://localhost:5432/themis",
            blob_root_dir=str(tmp_path / "blobs"),
        )
    )

    assert bundle.event_repo.__class__ is SqliteEventRepository
    assert bundle.projection_repo.__class__ is SqliteProjectionRepository
    assert bundle.observability_store.__class__ is SqliteObservabilityStore


def test_postgres_connection_manager_returns_explicit_storage_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, tuple[object, ...]]] = []

    class FakeCursor:
        def fetchone(self) -> None:
            return None

        def fetchall(self) -> list[object]:
            return []

    class FakeRawConnection:
        entered = False
        exited = False

        def execute(self, query: str, params: tuple[object, ...] = ()) -> FakeCursor:
            executed.append((query, params))
            return FakeCursor()

        def __enter__(self) -> FakeRawConnection:
            self.entered = True
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            self.exited = True
            return None

        def secret_method(self) -> str:
            return "not part of the storage contract"

    class FakeRows:
        dict_row = object()

    class FakePsycopgModule:
        rows = FakeRows()

        @staticmethod
        def connect(database_url: str, *, row_factory: object) -> FakeRawConnection:
            assert database_url == "postgresql://localhost:5432/themis"
            assert row_factory is FakeRows.dict_row
            return FakeRawConnection()

    monkeypatch.setattr(
        "themis.storage.postgres.manager.import_optional",
        lambda module_name, *, extra: FakePsycopgModule,
    )

    manager = PostgresConnectionManager("postgresql://localhost:5432/themis")

    with manager.get_connection() as conn:
        assert isinstance(conn, StorageConnection)
        conn.execute("SELECT ?, ?", ("a", 1))
        with pytest.raises(AttributeError):
            getattr(conn, "executescript")("SELECT 1; ; SELECT 2;")
        with pytest.raises(AttributeError):
            getattr(conn, "secret_method")()

    assert executed == [
        ("SELECT %s, %s", ("a", 1)),
    ]


def test_postgres_connection_manager_reopens_closed_cached_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_calls: list[int] = []

    class FakeCursor:
        def fetchone(self) -> None:
            return None

        def fetchall(self) -> list[object]:
            return []

    class FakeRawConnection:
        def __init__(self, index: int) -> None:
            self.index = index
            self.closed = False

        def execute(self, query: str, params: tuple[object, ...] = ()) -> FakeCursor:
            if self.closed:
                raise AssertionError("stale closed connection reused")
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True
            return None

    class FakeRows:
        dict_row = object()

    class FakePsycopgModule:
        rows = FakeRows()

        @staticmethod
        def connect(database_url: str, *, row_factory: object) -> FakeRawConnection:
            assert database_url == "postgresql://localhost:5432/themis"
            assert row_factory is FakeRows.dict_row
            connect_calls.append(len(connect_calls) + 1)
            return FakeRawConnection(connect_calls[-1])

    monkeypatch.setattr(
        "themis.storage.postgres.manager.import_optional",
        lambda module_name, *, extra: FakePsycopgModule,
    )

    manager = PostgresConnectionManager("postgresql://localhost:5432/themis")

    with manager.get_connection() as conn:
        with conn:
            conn.execute("SELECT 1")

    with manager.get_connection() as conn:
        conn.execute("SELECT 1")

    assert connect_calls == [1, 2]
