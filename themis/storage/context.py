"""Shared infrastructure context for storage sub-modules."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

from themis.storage.database import MetadataStore
from themis.storage.filesystem import FileSystem
from themis.storage.locking import LockManager
from themis.storage.models import StorageConfig


@dataclass
class StorageContext:
    """Shared infrastructure passed to each storage sub-module.

    This avoids duplicating filesystem, locking, and config references
    across sub-modules while keeping each module independently testable.
    """

    root: Path
    experiments_dir: Path
    config: StorageConfig
    fs: FileSystem
    lock_manager: LockManager
    metadata_store: MetadataStore | None

    # In-memory caches (shared mutable state)
    task_index: dict[str, set[str]] = field(default_factory=dict)
    template_index: dict[str, dict[str, str]] = field(default_factory=dict)
    run_dir_index: dict[str, Path] = field(default_factory=dict)

    @contextlib.contextmanager
    def acquire_lock(self, run_id: str):
        """Acquire exclusive lock for run directory."""
        run_dir = self.get_run_dir(run_id)
        lock_path = run_dir / ".lock"
        with self.lock_manager.acquire(run_id, lock_path):
            yield

    def get_run_dir(self, run_id: str, experiment_id: str | None = None) -> Path:
        """Get run directory path."""
        if experiment_id is not None:
            run_dir = self.experiments_dir / experiment_id / "runs" / run_id
            self.run_dir_index[run_id] = run_dir
            return run_dir

        cached_dir = self.run_dir_index.get(run_id)
        if cached_dir is not None and cached_dir.exists():
            return cached_dir

        if self.metadata_store:
            exp_id = self.metadata_store.get_run_experiment_id(run_id)
            if exp_id:
                indexed_dir = self.experiments_dir / exp_id / "runs" / run_id
                self.run_dir_index[run_id] = indexed_dir
                return indexed_dir

        scanned_dir = self._get_run_dir_by_scanning(run_id)
        if scanned_dir is not None:
            self.run_dir_index[run_id] = scanned_dir
            return scanned_dir

        # Default location for new runs
        return self.experiments_dir / "default" / "runs" / run_id

    def _get_run_dir_by_scanning(self, run_id: str) -> Path | None:
        """Resolve run path by scanning experiment directories."""
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue
            candidate_path = runs_dir / run_id / "metadata.json"
            if candidate_path.exists():
                return runs_dir / run_id
        return None

    def get_generation_dir(self, run_id: str) -> Path:
        """Get generation data directory."""
        return self.get_run_dir(run_id) / "generation"

    def get_evaluation_dir(self, run_id: str, eval_id: str = "default") -> Path:
        """Get evaluation directory."""
        return self.get_run_dir(run_id) / "evaluations" / eval_id
