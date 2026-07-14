from __future__ import annotations

import pytest

from themis.core.stores.postgres import postgres_store


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
