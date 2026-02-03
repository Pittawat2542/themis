from __future__ import annotations

from pathlib import Path

from themis.specs import StorageSpec


def test_storage_spec_defaults():
    spec = StorageSpec()
    assert spec.backend is None
    assert spec.path is None
    assert spec.cache is True


def test_storage_spec_path():
    spec = StorageSpec(path=Path(".cache/experiments"))
    assert spec.path == Path(".cache/experiments")
