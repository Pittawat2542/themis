from __future__ import annotations

from pathlib import Path

from themis.experiment.cache_manager import CacheManager


class _StorageWithPath:
    def __init__(self, path: Path):
        self._path = path

    def get_run_path(self, run_id: str) -> Path:
        return self._path / run_id


class _StorageWithoutPath:
    def get_run_path(self, run_id: str) -> None:
        return None


def test_get_run_path_returns_none_when_storage_returns_none():
    manager = CacheManager(storage=_StorageWithoutPath())

    assert manager.get_run_path("run-1") is None


def test_get_run_path_serializes_path_when_available(tmp_path: Path):
    manager = CacheManager(storage=_StorageWithPath(tmp_path))

    assert manager.get_run_path("run-1") == str(tmp_path / "run-1")
