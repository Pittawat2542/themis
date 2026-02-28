"""Main storage interface for experiment data."""

from __future__ import annotations

import contextlib
import hashlib
import json
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Iterable
from typing import Any
from dataclasses import asdict  # Added for StorageBackend interface

from themis.exceptions import StorageError

from themis.core import entities as core_entities
from themis.core import serialization as core_serialization
from themis.storage.cache_keys import evaluation_cache_key, task_cache_key
from themis.storage.database import MetadataStore
from themis.storage.filesystem import FileSystem
from themis.storage.locking import LockManager
from themis.storage.models import (
    RetentionPolicy,
    RunMetadata,
    RunStatus,
    StorageConfig,
)
from themis.backends.storage import StorageBackend


class ExperimentStorage(StorageBackend):
    """Robust storage with lifecycle management, locking, and integrity checks."""

    def __init__(self, root: str | Path, config: StorageConfig | None = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._config = config or StorageConfig()

        # Create experiments directory
        self._experiments_dir = self._root / "experiments"
        self._experiments_dir.mkdir(exist_ok=True)

        # Initialize helper components
        self._fs = FileSystem(self._config)
        self._lock_manager = LockManager()

        self._metadata_store = None
        if self._config.use_sqlite_metadata:
            self._metadata_store = MetadataStore(self._root / "experiments.db")

        # In-memory caches
        self._task_index: dict[str, set[str]] = {}
        self._template_index: dict[str, dict[str, str]] = {}
        self._run_dir_index: dict[str, Path] = {}

    @property
    def __repr__(self) -> str:
        return f"ExperimentStorage(root={self._root})"

    # === StorageBackend Interface Implementation ===

    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Save run metadata (StorageBackend API)."""
        experiment_id = metadata.get("experiment_id", "default")
        if not self.run_metadata_exists(run_id):
            self.start_run(
                run_id,
                experiment_id=experiment_id,
                config=metadata,
            )
            return

        run_metadata = self._load_run_metadata(run_id)
        run_metadata.experiment_id = experiment_id
        run_metadata.config_snapshot = dict(metadata)
        status = metadata.get("status")
        if status is not None:
            run_metadata.status = RunStatus(status)
        if "error_message" in metadata:
            run_metadata.error_message = metadata.get("error_message")
        run_metadata.updated_at = metadata.get("updated_at", run_metadata.updated_at)
        self._save_run_metadata(run_metadata)

    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Load run metadata (StorageBackend API)."""
        metadata = self._load_run_metadata(run_id)
        payload = asdict(metadata)
        payload["status"] = metadata.status.value
        return payload

    def save_generation_record(
        self, run_id: str, record: core_entities.GenerationRecord
    ) -> None:
        """Save generation record (StorageBackend API)."""
        if not self.run_metadata_exists(run_id):
            self.start_run(run_id, experiment_id="default")
        self.append_record(run_id, record)

    def load_generation_records(
        self, run_id: str
    ) -> list[core_entities.GenerationRecord]:
        """Load generation records (StorageBackend API)."""
        cached = self.load_cached_records(run_id)
        return list(cached.values())

    def save_evaluation_record(
        self,
        run_id: str,
        generation_record: core_entities.GenerationRecord,
        record: core_entities.EvaluationRecord,
    ) -> None:
        """Save evaluation record (StorageBackend API)."""
        if not self.run_metadata_exists(run_id):
            self.start_run(run_id, experiment_id="default")
        self.append_evaluation(run_id, generation_record, record)

    def load_evaluation_records(
        self, run_id: str
    ) -> dict[str, core_entities.EvaluationRecord]:
        """Load evaluation records (StorageBackend API)."""
        return self.load_cached_evaluations(run_id)

    def list_run_ids(self) -> list[str]:
        """List all run IDs in storage (StorageBackend API)."""
        return [run.run_id for run in self.list_runs()]

    def run_exists(self, run_id: str) -> bool:
        """Check if run exists (StorageBackend API)."""
        return self.run_metadata_exists(run_id)

    def save_report(self, run_id: str, report: core_entities.ExperimentReport) -> None:
        """Save report (StorageBackend API)."""
        if not self.run_metadata_exists(run_id):
            self.start_run(run_id, experiment_id="default")
        report_path = self.get_run_path(run_id) / "report.pkl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("wb") as handle:
            pickle.dump(report, handle)

        metadata_path = self.get_run_path(run_id) / "report_metadata.json"
        metadata = {
            "run_id": run_id,
            "generation_results": len(report.generation_results),
            "failures": len(report.failures),
            "report_metadata": report.metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_report(self, run_id: str) -> core_entities.ExperimentReport:
        """Load report (StorageBackend API)."""
        report_path = self.get_run_path(run_id) / "report.pkl"
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found for run {run_id}")
        with report_path.open("rb") as handle:
            report = pickle.load(handle)
        if not isinstance(report, core_entities.ExperimentReport):
            raise StorageError(f"Invalid report payload for run {run_id}")
        return report

    @property
    def root(self) -> Path:
        """Filesystem root directory for this storage instance."""
        return self._root

    @contextlib.contextmanager
    def _acquire_lock(self, run_id: str):
        """Acquire exclusive lock for run directory."""
        # Need to resolve run_dir to place lock file
        # We can't use _get_run_dir potentially because it might need metadata/lock?
        # Actually _get_run_dir logic relies on cached index or scanning.
        # For new runs, start_run defines the path.
        # For existing runs, we need a path.
        # Locking is per-run-id.
        # We'll use a standardized lock path relative to storage root for robustness
        # OR stick to the directory. Sticking to directory is better for cleanup.
        # If we don't know the directory yet, we might fallback to default?
        # But `start_run` passes experiment_id so we know the path.
        # `append_record` implies run exists.
        run_dir = self._get_run_dir(run_id)
        lock_path = run_dir / ".lock"
        with self._lock_manager.acquire(run_id, lock_path):
            yield

    def start_run(
        self,
        run_id: str,
        experiment_id: str,
        config: dict | None = None,
    ) -> RunMetadata:
        """Start a new run with in_progress status."""
        with self._acquire_lock(run_id):
            # Check if run already exists
            if self._run_metadata_exists(run_id):
                raise StorageError(f"Run {run_id} already exists")

            # Create run directory
            run_dir = self._get_run_dir(run_id, experiment_id=experiment_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create metadata
            metadata = RunMetadata(
                run_id=run_id,
                experiment_id=experiment_id,
                status=RunStatus.IN_PROGRESS,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                config_snapshot=config or {},
            )

            # Save metadata
            self._save_run_metadata(metadata)

            return metadata

    def complete_run(self, run_id: str):
        """Mark run as completed."""
        with self._acquire_lock(run_id):
            metadata = self._load_run_metadata(run_id)
            metadata.status = RunStatus.COMPLETED
            metadata.completed_at = datetime.now().isoformat()
            metadata.updated_at = datetime.now().isoformat()
            self._save_run_metadata(metadata)

    def fail_run(self, run_id: str, error_message: str):
        """Mark run as failed with error message."""
        with self._acquire_lock(run_id):
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
    ):
        """Update run progress counters."""
        with self._acquire_lock(run_id):
            metadata = self._load_run_metadata(run_id)

            if total_samples is not None:
                metadata.total_samples = total_samples
            if successful_generations is not None:
                metadata.successful_generations = successful_generations
            if failed_generations is not None:
                metadata.failed_generations = failed_generations

            metadata.updated_at = datetime.now().isoformat()
            self._save_run_metadata(metadata)

    def append_record(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        *,
        cache_key: str | None = None,
    ) -> None:
        """Append record with atomic write and locking."""
        with self._acquire_lock(run_id):
            # Ensure generation directory exists
            gen_dir = self._get_generation_dir(run_id)
            gen_dir.mkdir(parents=True, exist_ok=True)

            path = gen_dir / "records.jsonl"

            # Initialize file with header if needed
            if not self._fs.file_exists_any_compression(path):
                self._fs.write_jsonl_with_header(path, [], file_type="records")

            # Serialize record
            payload = self._serialize_record(run_id, record)
            payload["cache_key"] = cache_key or self._task_cache_key(record.task)

            # Atomic append
            self._fs.atomic_append(path, payload)

            # Update progress
            metadata = self._load_run_metadata(run_id)
            new_successful = metadata.successful_generations + (
                1 if record.output else 0
            )
            new_failed = metadata.failed_generations + (1 if record.error else 0)

            self.update_run_progress(
                run_id,
                total_samples=metadata.total_samples + 1,
                successful_generations=new_successful,
                failed_generations=new_failed,
            )

            # Auto-checkpoint if configured
            if self._config.checkpoint_interval > 0:
                total = new_successful + new_failed
                if total % self._config.checkpoint_interval == 0:
                    checkpoint_data = {
                        "total_samples": total,
                        "successful": new_successful,
                        "failed": new_failed,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.save_checkpoint(run_id, checkpoint_data)

    def _save_run_metadata(self, metadata: RunMetadata):
        """Save run metadata to both JSON and SQLite."""
        # Save to JSON file
        metadata_path = self._get_run_dir(metadata.run_id) / "metadata.json"
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

        # Save to SQLite
        if self._metadata_store:
            self._metadata_store.save_run_metadata(metadata)

    def _load_run_metadata(self, run_id: str) -> RunMetadata:
        """Load run metadata from JSON file."""
        metadata_path = self._get_run_dir(run_id) / "metadata.json"
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
        metadata_path = self._get_run_dir(run_id) / "metadata.json"
        return metadata_path.exists()

    def run_metadata_exists(self, run_id: str) -> bool:
        """Check if run metadata exists (public API)."""
        return self._run_metadata_exists(run_id)

    def _get_run_dir(self, run_id: str, experiment_id: str | None = None) -> Path:
        """Get run directory path."""
        if experiment_id is not None:
            run_dir = self._experiments_dir / experiment_id / "runs" / run_id
            self._run_dir_index[run_id] = run_dir
            return run_dir

        cached_dir = self._run_dir_index.get(run_id)
        if cached_dir is not None and cached_dir.exists():
            return cached_dir

        if self._metadata_store:
            exp_id = self._metadata_store.get_run_experiment_id(run_id)
            if exp_id:
                indexed_dir = self._experiments_dir / exp_id / "runs" / run_id
                self._run_dir_index[run_id] = indexed_dir
                return indexed_dir

        scanned_dir = self._get_run_dir_by_scanning(run_id)
        if scanned_dir is not None:
            self._run_dir_index[run_id] = scanned_dir
            return scanned_dir

        # Default location for new runs
        return self._experiments_dir / "default" / "runs" / run_id

    def _get_run_dir_by_scanning(self, run_id: str) -> Path | None:
        """Resolve run path by scanning experiment directories."""
        for exp_dir in self._experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue
            candidate_path = runs_dir / run_id / "metadata.json"
            if candidate_path.exists():
                return runs_dir / run_id
        return None

    def _get_generation_dir(self, run_id: str) -> Path:
        """Get generation data directory."""
        return self._get_run_dir(run_id) / "generation"

    def _get_evaluation_dir(self, run_id: str, eval_id: str = "default") -> Path:
        """Get evaluation directory."""
        return self._get_run_dir(run_id) / "evaluations" / eval_id

    def cache_dataset(self, run_id: str, dataset: Iterable[dict[str, object]]) -> None:
        """Cache dataset samples to storage."""
        if not self._config.save_dataset:
            return

        with self._acquire_lock(run_id):
            gen_dir = self._get_generation_dir(run_id)
            gen_dir.mkdir(parents=True, exist_ok=True)
            path = gen_dir / "dataset.jsonl"

            self._fs.write_jsonl_with_header(path, dataset, file_type="dataset")

    def load_dataset(self, run_id: str) -> list[dict[str, object]]:
        """Load cached dataset."""
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "dataset.jsonl"

        rows: list[dict[str, object]] = []
        with self._fs.open_for_read(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue
                rows.append(data)
        return rows

    def load_cached_records(
        self, run_id: str
    ) -> dict[str, core_entities.GenerationRecord]:
        """Load cached generation records."""
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "records.jsonl"

        try:
            handle = self._fs.open_for_read(path)
        except FileNotFoundError:
            return {}

        tasks = self._load_tasks(run_id)
        records: dict[str, core_entities.GenerationRecord] = {}

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                key = data.get("cache_key")
                if not key:
                    continue

                record = self._deserialize_record(data, tasks)
                records[key] = record

        return records

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
        with self._acquire_lock(run_id):
            eval_dir = self._get_evaluation_dir(run_id, eval_id)
            eval_dir.mkdir(parents=True, exist_ok=True)

            path = eval_dir / "evaluation.jsonl"

            if not self._fs.file_exists_any_compression(path):
                self._fs.write_jsonl_with_header(path, [], file_type="evaluation")

            # Use evaluation_cache_key that includes evaluation config
            cache_key = evaluation_cache_key(record.task, evaluation_config)

            payload = {
                "cache_key": cache_key,
                "evaluation": core_serialization.serialize_evaluation_record(
                    evaluation
                ),
            }
            self._fs.atomic_append(path, payload)

    def load_cached_evaluations(
        self,
        run_id: str,
        eval_id: str = "default",
        evaluation_config: dict | None = None,
    ) -> dict[str, core_entities.EvaluationRecord]:
        """Load cached evaluation records."""
        eval_dir = self._get_evaluation_dir(run_id, eval_id)
        path = eval_dir / "evaluation.jsonl"

        try:
            handle = self._fs.open_for_read(path)
        except FileNotFoundError:
            return {}

        evaluations: dict[str, core_entities.EvaluationRecord] = {}

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                key = data.get("cache_key")
                if not key:
                    continue

                # Check consistency if needed (implied by cache key usage in orchestrator)
                # Here we just load all and let orchestrator filter by key lookup

                evaluations[key] = core_serialization.deserialize_evaluation_record(
                    data["evaluation"]
                )

        return evaluations

    def get_run_path(self, run_id: str) -> Path:
        """Get the filesystem path for a run's storage directory."""
        return self._get_run_dir(run_id)

    def _serialize_record(
        self, run_id: str, record: core_entities.GenerationRecord
    ) -> dict:
        """Serialize generation record."""
        # This calls _persist_task which atomically writes task if needed
        task_key = self._persist_task(run_id, record.task)

        output_data = None
        if record.output:
            output_data = {"text": record.output.text}
            if self._config.save_raw_responses:
                output_data["raw"] = record.output.raw

        return {
            "task_key": task_key,
            "output": output_data,
            "error": {
                "message": record.error.message,
                "kind": record.error.kind,
                "details": record.error.details,
            }
            if record.error
            else None,
            "metrics": record.metrics,
            "attempts": [
                self._serialize_record(run_id, attempt) for attempt in record.attempts
            ],
        }

    def _deserialize_record(
        self, payload: dict, tasks: dict[str, core_entities.GenerationTask]
    ) -> core_entities.GenerationRecord:
        """Deserialize generation record."""
        task_key = payload["task_key"]
        task = tasks[task_key]
        output_data = payload.get("output")
        error_data = payload.get("error")

        record = core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(
                text=output_data["text"], raw=output_data.get("raw")
            )
            if output_data
            else None,
            error=core_entities.ModelError(
                message=error_data["message"],
                kind=error_data.get("kind", "model_error"),
                details=error_data.get("details", {}),
            )
            if error_data
            else None,
            metrics=payload.get("metrics", {}),
        )

        record.attempts = [
            self._deserialize_record(attempt, tasks)
            for attempt in payload.get("attempts", [])
        ]

        return record

    def _persist_task(self, run_id: str, task: core_entities.GenerationTask) -> str:
        """Persist task and return cache key."""
        key = self._task_cache_key(task)
        index = self._load_task_index(run_id)

        if key in index:
            return key

        gen_dir = self._get_generation_dir(run_id)
        gen_dir.mkdir(parents=True, exist_ok=True)
        path = gen_dir / "tasks.jsonl"

        if not self._fs.file_exists_any_compression(path):
            self._fs.write_jsonl_with_header(path, [], file_type="tasks")

        if self._config.deduplicate_templates:
            template_id = self._persist_template(run_id, task.prompt.spec)
            task_data = core_serialization.serialize_generation_task(task)
            task_data["prompt"]["spec"] = {"_template_ref": template_id}
        else:
            task_data = core_serialization.serialize_generation_task(task)

        payload = {"task_key": key, "task": task_data}
        self._fs.atomic_append(path, payload)

        index.add(key)
        self._save_task_index(run_id, index)

        return key

    def _persist_template(self, run_id: str, spec: core_entities.PromptSpec) -> str:
        """Persist prompt template."""
        template_content = f"{spec.name}:{spec.template}"
        template_id = hashlib.sha256(template_content.encode("utf-8")).hexdigest()[:16]

        if run_id not in self._template_index:
            self._template_index[run_id] = {}
            self._load_templates(run_id)

        if template_id in self._template_index[run_id]:
            return template_id

        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "templates.jsonl"

        if not self._fs.file_exists_any_compression(path):
            self._fs.write_jsonl_with_header(path, [], file_type="templates")

        payload = {
            "template_id": template_id,
            "spec": core_serialization.serialize_prompt_spec(spec),
        }
        self._fs.atomic_append(path, payload)

        self._template_index[run_id][template_id] = spec.template
        return template_id

    def _load_task_index(self, run_id: str) -> set[str]:
        """Load task index from disk cache."""
        if run_id in self._task_index:
            return self._task_index[run_id]

        index_path = self._get_run_dir(run_id) / ".index.json"
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
            self._task_index[run_id] = set(index_data.get("task_keys", []))
            return self._task_index[run_id]

        self._task_index[run_id] = set()
        return self._task_index[run_id]

    def _save_task_index(self, run_id: str, index: set[str]):
        """Save task index to disk."""
        index_path = self._get_run_dir(run_id) / ".index.json"
        index_data = {
            "task_keys": list(index),
            "template_ids": self._template_index.get(run_id, {}),
            "last_updated": datetime.now().isoformat(),
        }
        index_path.write_text(json.dumps(index_data))

    def _load_templates(self, run_id: str) -> dict[str, core_entities.PromptSpec]:
        """Load templates from disk."""
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "templates.jsonl"

        templates: dict[str, core_entities.PromptSpec] = {}
        try:
            handle = self._fs.open_for_read(path)
        except FileNotFoundError:
            return templates

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                template_id = data["template_id"]
                templates[template_id] = core_serialization.deserialize_prompt_spec(
                    data["spec"]
                )

        return templates

    def _load_tasks(self, run_id: str) -> dict[str, core_entities.GenerationTask]:
        """Load tasks from disk."""
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "tasks.jsonl"

        tasks: dict[str, core_entities.GenerationTask] = {}
        try:
            handle = self._fs.open_for_read(path)
        except FileNotFoundError:
            return tasks

        templates = (
            self._load_templates(run_id) if self._config.deduplicate_templates else {}
        )

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                task_key = data["task_key"]
                task_data = data["task"]

                if (
                    self._config.deduplicate_templates
                    and "_template_ref" in task_data.get("prompt", {}).get("spec", {})
                ):
                    template_id = task_data["prompt"]["spec"]["_template_ref"]
                    if template_id in templates:
                        task_data["prompt"]["spec"] = (
                            core_serialization.serialize_prompt_spec(
                                templates[template_id]
                            )
                        )

                tasks[task_key] = core_serialization.deserialize_generation_task(
                    task_data
                )

        self._task_index[run_id] = set(tasks.keys())
        return tasks

    def _task_cache_key(self, task: core_entities.GenerationTask) -> str:
        """Generate cache key for task."""
        return task_cache_key(task)

    def save_checkpoint(self, run_id: str, checkpoint_data: dict):
        """Save checkpoint for resumability."""
        with self._acquire_lock(run_id):
            checkpoint_dir = self._get_run_dir(run_id) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.json"

            checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

    def load_latest_checkpoint(self, run_id: str) -> dict | None:
        """Load most recent checkpoint."""
        checkpoint_dir = self._get_run_dir(run_id) / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
        if not checkpoints:
            return None

        return json.loads(checkpoints[0].read_text())

    def apply_retention_policy(self, policy: RetentionPolicy | None = None):
        """Apply retention policy to clean up old runs."""
        policy = policy or self._config.retention_policy
        if not policy:
            return

        for exp_dir in self._experiments_dir.iterdir():
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

    def _delete_run_dir(self, run_dir: Path):
        """Delete run directory and update database."""
        run_id = run_dir.name
        self._run_dir_index.pop(run_id, None)

        if self._metadata_store:
            self._metadata_store.delete_run(run_id)

        shutil.rmtree(run_dir, ignore_errors=True)

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its stored artifacts."""
        run_dir = self._get_run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        self._delete_run_dir(run_dir)

    def get_storage_size(self, experiment_id: str | None = None) -> int:
        """Get total storage size in bytes."""
        if experiment_id:
            exp_dir = self._experiments_dir / experiment_id
            if not exp_dir.exists():
                return 0
            return sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file())
        else:
            return sum(
                f.stat().st_size
                for f in self._experiments_dir.rglob("*")
                if f.is_file()
            )

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        if self._metadata_store:
            return self._metadata_store.list_runs(experiment_id, status, limit)
        return self._list_runs_from_files(experiment_id, status, limit)

    def _list_runs_from_files(
        self, experiment_id: str | None, status: RunStatus | None, limit: int | None
    ) -> list[RunMetadata]:
        """List runs by scanning files (fallback)."""
        runs = []

        exp_dirs = (
            [self._experiments_dir / experiment_id]
            if experiment_id
            else list(self._experiments_dir.iterdir())
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
