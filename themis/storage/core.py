"""Main storage interface for experiment data.

ExperimentStorage is a backwards-compatible facade that delegates to
focused sub-modules:

- ``StorageContext``: shared infrastructure (dirs, locking, caches)
- ``RunLifecycleManager``: start/complete/fail, metadata, checkpoints, admin
- ``RecordSerializer``: record serialization, task/template persistence
- ``GenerationIO``: generation record append/load, dataset caching
- ``EvaluationIO``: evaluation record append/load
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from dataclasses import asdict


from themis.core import entities as core_entities
from themis.storage.cache_keys import evaluation_cache_key, task_cache_key  # noqa: F401
from themis.storage.context import StorageContext
from themis.storage.database import MetadataStore
from themis.storage.filesystem import FileSystem
from themis.storage.locking import LockManager
from themis.storage.models import (
    RetentionPolicy,
    RunMetadata,
    RunStatus,
    StorageConfig,
)
from themis.storage.run_lifecycle import RunLifecycleManager
from themis.storage.serialization import RecordSerializer
from themis.storage.generation_io import GenerationIO
from themis.storage.evaluation_io import EvaluationIO
from themis.backends.storage import StorageBackend


class ExperimentStorage(StorageBackend):
    """Robust storage with lifecycle management, locking, and integrity checks.

    This class is a facade that delegates to focused sub-modules for
    run lifecycle, generation I/O, evaluation I/O, and serialization.
    """

    def __init__(self, root: str | Path, config: StorageConfig | None = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._config = config or StorageConfig()

        experiments_dir = self._root / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        metadata_store = None
        if self._config.use_sqlite_metadata:
            metadata_store = MetadataStore(self._root / "experiments.db")

        # Shared context for all sub-modules
        self._ctx = StorageContext(
            root=self._root,
            experiments_dir=experiments_dir,
            config=self._config,
            fs=FileSystem(self._config),
            lock_manager=LockManager(),
            metadata_store=metadata_store,
        )

        # Sub-modules
        self._lifecycle = RunLifecycleManager(self._ctx)
        self._serializer = RecordSerializer(self._ctx)
        self._gen_io = GenerationIO(self._ctx, self._serializer)
        self._eval_io = EvaluationIO(self._ctx)

    def __repr__(self) -> str:
        return f"ExperimentStorage(root={self._root})"

    @property
    def root(self) -> Path:
        """Filesystem root directory for this storage instance."""
        return self._root

    # ==================================================================
    # StorageBackend API — thin delegates
    # ==================================================================

    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Save run metadata (StorageBackend API)."""
        run_meta = RunMetadata(
            run_id=run_id,
            experiment_id=metadata.get("experiment_id", "default"),
            status=RunStatus(metadata.get("status", "in_progress")),
            created_at=metadata.get("created_at", ""),
            updated_at=metadata.get("updated_at", ""),
            completed_at=metadata.get("completed_at"),
            total_samples=metadata.get("total_samples", 0),
            successful_generations=metadata.get("successful_generations", 0),
            failed_generations=metadata.get("failed_generations", 0),
            config_snapshot=metadata.get("config_snapshot", {}),
            error_message=metadata.get("error_message"),
        )
        self._lifecycle._save_run_metadata(run_meta)

    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Load run metadata (StorageBackend API)."""
        meta = self._lifecycle._load_run_metadata(run_id)
        return asdict(meta)

    def save_generation_record(
        self, run_id: str, record: core_entities.GenerationRecord
    ) -> None:
        """Save generation record (StorageBackend API)."""
        self._gen_io.append_record(run_id, record)

    def load_generation_records(
        self, run_id: str
    ) -> dict[str, core_entities.GenerationRecord]:
        """Load generation records (StorageBackend API)."""
        return self._gen_io.load_cached_records(run_id)

    def save_evaluation_record(
        self,
        run_id: str,
        generation_record: core_entities.GenerationRecord,
        record: core_entities.EvaluationRecord,
    ) -> None:
        """Save evaluation record (StorageBackend API)."""
        if not self._lifecycle.run_metadata_exists(run_id):
            self.start_run(run_id, experiment_id="default")
        self._eval_io.append_evaluation(run_id, generation_record, record)

    def append_evaluation_record(
        self,
        run_id: str,
        generation_record: core_entities.GenerationRecord,
        evaluation_record: core_entities.EvaluationRecord,
        *,
        evaluation_config: dict | None = None,
    ) -> None:
        """Append evaluation record, preserving evaluation_config for cache key generation."""
        if not self._lifecycle.run_metadata_exists(run_id):
            self.start_run(run_id, experiment_id="default")
        self._eval_io.append_evaluation(
            run_id,
            generation_record,
            evaluation_record,
            evaluation_config=evaluation_config,
        )

    def load_evaluation_records(
        self, run_id: str
    ) -> dict[str, core_entities.EvaluationRecord]:
        """Load evaluation records (StorageBackend API)."""
        return self._eval_io.load_cached_evaluations(run_id)

    def list_run_ids(self) -> list[str]:
        """List all run IDs in storage (StorageBackend API)."""
        return [r.run_id for r in self._lifecycle.list_runs()]

    def run_exists(self, run_id: str) -> bool:
        """Check if run exists (StorageBackend API)."""
        return self._lifecycle.run_metadata_exists(run_id)

    def save_report(self, run_id: str, report: core_entities.ExperimentReport) -> None:
        """Save report (StorageBackend API)."""
        run_dir = self._ctx.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        report_path = run_dir / "report.pkl"
        with report_path.open("wb") as f:
            pickle.dump(report, f)

        try:
            report_json_path = run_dir / "report_summary.json"
            summary = {
                "run_id": run_id,
                "total_records": len(report.generation_results),
                "metadata": report.metadata,
            }
            report_json_path.write_text(json.dumps(summary, indent=2, default=str))
        except Exception:
            pass

    def load_report(self, run_id: str) -> core_entities.ExperimentReport | None:
        """Load report (StorageBackend API)."""
        run_dir = self._ctx.get_run_dir(run_id)
        report_path = run_dir / "report.pkl"
        if not report_path.exists():
            return None
        with report_path.open("rb") as f:
            return pickle.load(f)

    # ==================================================================
    # Lifecycle delegates
    # ==================================================================

    def start_run(
        self,
        run_id: str,
        experiment_id: str,
        config: dict | None = None,
    ) -> RunMetadata:
        """Start a new run with in_progress status."""
        return self._lifecycle.start_run(run_id, experiment_id, config)

    def complete_run(self, run_id: str) -> None:
        """Mark run as completed."""
        self._lifecycle.complete_run(run_id)

    def fail_run(self, run_id: str, error_message: str) -> None:
        """Mark run as failed with error message."""
        self._lifecycle.fail_run(run_id, error_message)

    def update_run_progress(
        self,
        run_id: str,
        total_samples: int | None = None,
        successful_generations: int | None = None,
        failed_generations: int | None = None,
    ) -> None:
        """Update run progress counters."""
        self._lifecycle.update_run_progress(
            run_id, total_samples, successful_generations, failed_generations
        )

    def run_metadata_exists(self, run_id: str) -> bool:
        """Check if run metadata exists (public API)."""
        return self._lifecycle.run_metadata_exists(run_id)

    # ==================================================================
    # Generation I/O delegates
    # ==================================================================

    def append_record(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        *,
        cache_key: str | None = None,
    ) -> None:
        """Append record with atomic write and locking."""
        self._gen_io.append_record(run_id, record, cache_key=cache_key)

        # Update progress (kept here since it crosses lifecycle boundary)
        metadata = self._lifecycle._load_run_metadata(run_id)
        new_successful = metadata.successful_generations + (1 if record.output else 0)
        new_failed = metadata.failed_generations + (1 if record.error else 0)
        self._lifecycle.update_run_progress(
            run_id,
            total_samples=metadata.total_samples + 1,
            successful_generations=new_successful,
            failed_generations=new_failed,
        )

        # Auto-checkpoint if configured
        from datetime import datetime

        if self._config.checkpoint_interval > 0:
            total = new_successful + new_failed
            if total % self._config.checkpoint_interval == 0:
                checkpoint_data = {
                    "total_samples": total,
                    "successful": new_successful,
                    "failed": new_failed,
                    "timestamp": datetime.now().isoformat(),
                }
                self._lifecycle.save_checkpoint(run_id, checkpoint_data)

    def load_cached_records(
        self, run_id: str
    ) -> dict[str, core_entities.GenerationRecord]:
        """Load cached generation records."""
        return self._gen_io.load_cached_records(run_id)

    def cache_dataset(self, run_id: str, dataset: Iterable[dict[str, object]]) -> None:
        """Cache dataset samples to storage."""
        self._gen_io.cache_dataset(run_id, dataset)

    def load_dataset(self, run_id: str) -> list[dict[str, object]]:
        """Load cached dataset."""
        return self._gen_io.load_dataset(run_id)

    # ==================================================================
    # Evaluation I/O delegates
    # ==================================================================

    def append_evaluation(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        evaluation: core_entities.EvaluationRecord,
        *,
        eval_id: str = "default",
        evaluation_config: dict | None = None,
    ) -> None:
        """Append evaluation result."""
        self._eval_io.append_evaluation(
            run_id,
            record,
            evaluation,
            eval_id=eval_id,
            evaluation_config=evaluation_config,
        )

    def load_cached_evaluations(
        self,
        run_id: str,
        eval_id: str = "default",
        evaluation_config: dict | None = None,
    ) -> dict[str, core_entities.EvaluationRecord]:
        """Load cached evaluation records."""
        return self._eval_io.load_cached_evaluations(run_id, eval_id, evaluation_config)

    # ==================================================================
    # Path helpers and admin
    # ==================================================================

    def get_run_path(self, run_id: str) -> Path:
        """Get the filesystem path for a run's storage directory."""
        return self._ctx.get_run_dir(run_id)

    def save_checkpoint(self, run_id: str, checkpoint_data: dict) -> None:
        """Save checkpoint for resumability."""
        self._lifecycle.save_checkpoint(run_id, checkpoint_data)

    def load_latest_checkpoint(self, run_id: str) -> dict | None:
        """Load most recent checkpoint."""
        return self._lifecycle.load_latest_checkpoint(run_id)

    def apply_retention_policy(self, policy: RetentionPolicy | None = None) -> None:
        """Apply retention policy to clean up old runs."""
        self._lifecycle.apply_retention_policy(policy)

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its stored artifacts."""
        self._lifecycle.delete_run(run_id)

    def get_storage_size(self, experiment_id: str | None = None) -> int:
        """Get total storage size in bytes."""
        return self._lifecycle.get_storage_size(experiment_id)

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        return self._lifecycle.list_runs(experiment_id, status, limit)

    def _get_run_dir(self, run_id: str, experiment_id: str | None = None) -> Path:
        return self._ctx.get_run_dir(run_id, experiment_id)

    def _get_generation_dir(self, run_id: str) -> Path:
        return self._ctx.get_generation_dir(run_id)

    def _get_evaluation_dir(self, run_id: str, eval_id: str = "default") -> Path:
        return self._ctx.get_evaluation_dir(run_id, eval_id)

    def _acquire_lock(self, run_id: str):
        return self._ctx.acquire_lock(run_id)

    def _load_run_metadata(self, run_id: str) -> RunMetadata:
        return self._lifecycle._load_run_metadata(run_id)

    def _run_metadata_exists(self, run_id: str) -> bool:
        return self._lifecycle._run_metadata_exists(run_id)

    def _task_cache_key(self, task: core_entities.GenerationTask) -> str:
        return task_cache_key(task)

    def _save_run_metadata(self, metadata: RunMetadata) -> None:
        return self._lifecycle._save_run_metadata(metadata)

    def _load_templates(self, run_id: str):
        return self._serializer.load_templates(run_id)

    def _load_tasks(self, run_id: str):
        return self._serializer.load_tasks(run_id)

    def _serialize_record(self, run_id: str, record: core_entities.GenerationRecord):
        return self._serializer.serialize_record(run_id, record)

    def _deserialize_record(self, payload: dict, tasks: dict):
        return self._serializer.deserialize_record(payload, tasks)

    def _persist_task(self, run_id: str, task: core_entities.GenerationTask):
        return self._serializer.persist_task(run_id, task)

    @property
    def _lock_manager(self):
        return self._ctx.lock_manager

    @property
    def _experiments_dir(self):
        return self._ctx.experiments_dir

    @property
    def _fs(self):
        return self._ctx.fs

    @property
    def _metadata_store(self):
        return self._ctx.metadata_store

    @property
    def _task_index(self):
        return self._ctx.task_index

    @property
    def _template_index(self):
        return self._ctx.template_index

    @property
    def _run_dir_index(self):
        return self._ctx.run_dir_index
