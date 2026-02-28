"""Storage backends and adapters for vNext workflows."""

from pathlib import Path
from typing import Any
from themis.storage.core import (
    ExperimentStorage,
    evaluation_cache_key,
    task_cache_key,
)
from themis.storage.filesystem import STORAGE_FORMAT_VERSION
from themis.storage.models import (
    ConcurrentAccessError,
    DataIntegrityError,
    RetentionPolicy,
    RunMetadata,
    RunStatus,
    StorageConfig,
)
from themis.exceptions import ConfigurationError

__all__ = [
    "ExperimentStorage",
    "RunStatus",
    "RunMetadata",
    "StorageConfig",
    "RetentionPolicy",
    "DataIntegrityError",
    "ConcurrentAccessError",
    "task_cache_key",
    "evaluation_cache_key",
    "STORAGE_FORMAT_VERSION",
]


def resolve_storage(
    storage_path: str | Path | None, storage_backend: Any | None = None
):
    """Resolve storage backend from path or explicit backend."""
    if storage_backend is not None:
        backend = storage_backend
        if hasattr(backend, "experiment_storage"):
            return backend.experiment_storage
        if not hasattr(backend, "start_run"):
            raise ConfigurationError(
                "storage_backend must be ExperimentStorage-compatible."
            )
        return backend

    if storage_path is None:
        return None

    root = Path(storage_path)
    return ExperimentStorage(root)
