from __future__ import annotations

import pytest

from themis.core.config import StorageConfig
from themis.core.stores import create_run_store
from themis.core.stores.postgres import PostgresRunStore, postgres_store


def test_store_factory_can_build_postgres_backend(tmp_path) -> None:
    store = create_run_store(
        StorageConfig(
            store="postgres",
            parameters={
                "url": f"postgresql://localhost/{tmp_path.name}",
                "blob_root": str(tmp_path / "postgres-blobs"),
            },
        )
    )

    assert isinstance(store, PostgresRunStore)


def test_postgres_store_raises_clear_import_error_when_dependency_is_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "themis.core.stores.postgres.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    store = postgres_store("postgresql://localhost/themis", tmp_path / "postgres-blobs")

    with pytest.raises(ImportError, match="optional 'postgres' dependency"):
        store.initialize()
