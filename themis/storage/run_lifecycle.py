"""Run lifecycle management: start, complete, fail, progress, metadata."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from themis.exceptions import StorageError
from themis.storage.context import StorageContext
from themis.storage.models import RetentionPolicy, RunMetadata, RunStatus


class RunLifecycleManager:
    """Manages the lifecycle of experiment runs in storage."""

    def __init__(self, ctx: StorageContext) -> None:
        self._ctx = ctx

    def start_run(
        self,
        run_id: str,
        experiment_id: str,
        config: dict | None = None,
    ) -> RunMetadata:
        """Start a new run with in_progress status."""
        with self._ctx.acquire_lock(run_id):
            if self._run_metadata_exists(run_id):
                raise StorageError(f"Run {run_id} already exists")

            run_dir = self._ctx.get_run_dir(run_id, experiment_id=experiment_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            metadata = RunMetadata(
                run_id=run_id,
                experiment_id=experiment_id,
                status=RunStatus.IN_PROGRESS,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                config_snapshot=config or {},
            )
            self._save_run_metadata(metadata)
            return metadata

    def complete_run(self, run_id: str) -> None:
        """Mark run as completed."""
        with self._ctx.acquire_lock(run_id):
            metadata = self._load_run_metadata(run_id)
            metadata.status = RunStatus.COMPLETED
            metadata.completed_at = datetime.now().isoformat()
            metadata.updated_at = datetime.now().isoformat()
            self._save_run_metadata(metadata)

    def fail_run(self, run_id: str, error_message: str) -> None:
        """Mark run as failed with error message."""
        with self._ctx.acquire_lock(run_id):
            metadata = self._load_run_metadata(run_id)
            metadata.status = RunStatus.FAILED
            metadata.error_message = error_message
            metadata.updated_at = datetime.now().isoformat()
            self._save_run_metadata(metadata)

    def update_run_progress(
        self,
        run_id: str,
        total_samples: int | None = None,
        successful_generations: int | None = None,
        failed_generations: int | None = None,
    ) -> None:
        """Update run progress counters."""
        with self._ctx.acquire_lock(run_id):
            metadata = self._load_run_metadata(run_id)
            if total_samples is not None:
                metadata.total_samples = total_samples
            if successful_generations is not None:
                metadata.successful_generations = successful_generations
            if failed_generations is not None:
                metadata.failed_generations = failed_generations
            metadata.updated_at = datetime.now().isoformat()
            self._save_run_metadata(metadata)

    # ---- Metadata CRUD ----

    def _save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata to both JSON and SQLite."""
        metadata_path = self._ctx.get_run_dir(metadata.run_id) / "metadata.json"
        metadata_dict = {
            "run_id": metadata.run_id,
            "experiment_id": metadata.experiment_id,
            "status": metadata.status.value,
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
            "completed_at": metadata.completed_at,
            "total_samples": metadata.total_samples,
            "successful_generations": metadata.successful_generations,
            "failed_generations": metadata.failed_generations,
            "config_snapshot": metadata.config_snapshot,
            "error_message": metadata.error_message,
        }
        metadata_path.write_text(json.dumps(metadata_dict, indent=2))

        if self._ctx.metadata_store:
            self._ctx.metadata_store.save_run_metadata(metadata)

    def _load_run_metadata(self, run_id: str) -> RunMetadata:
        """Load run metadata from JSON file."""
        metadata_path = self._ctx.get_run_dir(run_id) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run metadata not found for {run_id}")

        data = json.loads(metadata_path.read_text())
        return RunMetadata(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            status=RunStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            completed_at=data.get("completed_at"),
            total_samples=data.get("total_samples", 0),
            successful_generations=data.get("successful_generations", 0),
            failed_generations=data.get("failed_generations", 0),
            config_snapshot=data.get("config_snapshot", {}),
            error_message=data.get("error_message"),
        )

    def _run_metadata_exists(self, run_id: str) -> bool:
        """Check if run metadata exists."""
        metadata_path = self._ctx.get_run_dir(run_id) / "metadata.json"
        return metadata_path.exists()

    def run_metadata_exists(self, run_id: str) -> bool:
        """Check if run metadata exists (public API)."""
        return self._run_metadata_exists(run_id)

    # ---- Checkpoint management ----

    def save_checkpoint(self, run_id: str, checkpoint_data: dict) -> None:
        """Save checkpoint for resumability."""
        with self._ctx.acquire_lock(run_id):
            checkpoint_dir = self._ctx.get_run_dir(run_id) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.json"
            checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

    def load_latest_checkpoint(self, run_id: str) -> dict | None:
        """Load most recent checkpoint."""
        checkpoint_dir = self._ctx.get_run_dir(run_id) / "checkpoints"
        if not checkpoint_dir.exists():
            return None
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
        if not checkpoints:
            return None
        return json.loads(checkpoints[0].read_text())

    # ---- Admin: listing, deletion, retention ----

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        if self._ctx.metadata_store:
            return self._ctx.metadata_store.list_runs(experiment_id, status, limit)
        return self._list_runs_from_files(experiment_id, status, limit)

    def _list_runs_from_files(
        self,
        experiment_id: str | None,
        status: RunStatus | None,
        limit: int | None,
    ) -> list[RunMetadata]:
        """List runs by scanning files (fallback)."""
        runs: list[RunMetadata] = []
        exp_dirs = (
            [self._ctx.experiments_dir / experiment_id]
            if experiment_id
            else list(self._ctx.experiments_dir.iterdir())
        )

        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue

            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                metadata_path = run_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                try:
                    data = json.loads(metadata_path.read_text())
                    run_meta = RunMetadata(
                        run_id=data["run_id"],
                        experiment_id=data["experiment_id"],
                        status=RunStatus(data["status"]),
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                        completed_at=data.get("completed_at"),
                        total_samples=data.get("total_samples", 0),
                        successful_generations=data.get("successful_generations", 0),
                        failed_generations=data.get("failed_generations", 0),
                        config_snapshot=data.get("config_snapshot", {}),
                        error_message=data.get("error_message"),
                    )
                    if status and run_meta.status != status:
                        continue
                    runs.append(run_meta)
                except Exception:
                    continue

        runs.sort(key=lambda x: x.created_at, reverse=True)
        return runs[:limit] if limit else runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its stored artifacts."""
        run_dir = self._ctx.get_run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        self._delete_run_dir(run_dir)

    def _delete_run_dir(self, run_dir: Path) -> None:
        """Delete run directory and update database."""
        run_id = run_dir.name
        self._ctx.run_dir_index.pop(run_id, None)
        if self._ctx.metadata_store:
            self._ctx.metadata_store.delete_run(run_id)
        shutil.rmtree(run_dir, ignore_errors=True)

    def get_storage_size(self, experiment_id: str | None = None) -> int:
        """Get total storage size in bytes."""
        if experiment_id:
            exp_dir = self._ctx.experiments_dir / experiment_id
            if not exp_dir.exists():
                return 0
            return sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file())
        else:
            return sum(
                f.stat().st_size
                for f in self._ctx.experiments_dir.rglob("*")
                if f.is_file()
            )

    def apply_retention_policy(self, policy: RetentionPolicy | None = None) -> None:
        """Apply retention policy to clean up old runs."""
        policy = policy or self._ctx.config.retention_policy
        if not policy:
            return

        for exp_dir in self._ctx.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue

            runs = []
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                metadata_path = run_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                try:
                    metadata = self._load_run_metadata(run_dir.name)
                    runs.append((run_dir, metadata))
                except Exception:
                    continue

            runs.sort(key=lambda x: x[1].created_at, reverse=True)

            runs_to_delete = []
            for i, (run_dir, metadata) in enumerate(runs):
                if i < policy.keep_latest_n:
                    continue
                if (
                    policy.keep_completed_only
                    and metadata.status != RunStatus.COMPLETED
                ):
                    runs_to_delete.append(run_dir)
                    continue
                if policy.max_age_days:
                    created = datetime.fromisoformat(metadata.created_at)
                    age = datetime.now() - created
                    if age > timedelta(days=policy.max_age_days):
                        runs_to_delete.append(run_dir)
                        continue
                if policy.max_runs_per_experiment:
                    if i >= policy.max_runs_per_experiment:
                        runs_to_delete.append(run_dir)

            for run_dir in runs_to_delete:
                self._delete_run_dir(run_dir)
