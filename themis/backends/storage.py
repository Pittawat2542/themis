"""Storage backend interface for custom storage implementations.

This module defines the abstract interface for storage backends, allowing
users to implement custom storage solutions (cloud storage, databases, etc.)
without modifying Themis core code.

Example implementations:
- S3Backend: Store results in AWS S3
- GCSBackend: Store results in Google Cloud Storage
- PostgresBackend: Store results in PostgreSQL
- RedisBackend: Use Redis for distributed caching
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from themis.core.entities import (
    EvaluationRecord,
    ExperimentReport,
    GenerationRecord,
)


class StorageBackend(ABC):
    """Abstract interface for storage backends.

    Implement this interface to create custom storage solutions.
    All methods should be thread-safe if used with concurrent workers.

    Example:
        >>> class S3StorageBackend(StorageBackend):
        ...     def __init__(self, bucket: str):
        ...         self.bucket = bucket
        ...         self.s3_client = boto3.client('s3')
        ...
        ...     def save_run_metadata(self, run_id: str, metadata: RunMetadata) -> None:
        ...         key = f"runs/{run_id}/metadata.json"
        ...         self.s3_client.put_object(
        ...             Bucket=self.bucket,
        ...             Key=key,
        ...             Body=metadata.to_json(),
        ...         )
        ...     # ... implement other methods
    """

    @abstractmethod
    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Save run metadata.

        Args:
            run_id: Unique identifier for the run
            metadata: Run metadata to save (as dictionary)
        """
        pass

    @abstractmethod
    def load_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load run metadata.

        Args:
            run_id: Unique identifier for the run

        Returns:
            Run metadata as dictionary

        Raises:
            FileNotFoundError: If run metadata doesn't exist
        """
        pass

    @abstractmethod
    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        """Save a generation record.

        Args:
            run_id: Unique identifier for the run
            record: Generation record to save

        Note:
            This method should be atomic and thread-safe.
        """
        pass

    @abstractmethod
    def load_generation_records(self, run_id: str) -> List[GenerationRecord]:
        """Load all generation records for a run.

        Args:
            run_id: Unique identifier for the run

        Returns:
            List of generation records
        """
        pass

    @abstractmethod
    def save_evaluation_record(
        self,
        run_id: str,
        generation_record: GenerationRecord,
        record: EvaluationRecord,
    ) -> None:
        """Save an evaluation record.

        Args:
            run_id: Unique identifier for the run
            generation_record: Generation record corresponding to this evaluation
            record: Evaluation record to save

        Note:
            This method should be atomic and thread-safe.
        """
        pass

    @abstractmethod
    def load_evaluation_records(self, run_id: str) -> Dict[str, EvaluationRecord]:
        """Load all evaluation records for a run.

        Args:
            run_id: Unique identifier for the run

        Returns:
            Dictionary mapping cache_key to EvaluationRecord
        """
        pass

    @abstractmethod
    def save_report(self, run_id: str, report: ExperimentReport) -> None:
        """Save experiment report.

        Args:
            run_id: Unique identifier for the run
            report: Experiment report to save
        """
        pass

    @abstractmethod
    def load_report(self, run_id: str) -> ExperimentReport:
        """Load experiment report.

        Args:
            run_id: Unique identifier for the run

        Returns:
            Experiment report

        Raises:
            FileNotFoundError: If report doesn't exist
        """
        pass

    @abstractmethod
    def list_runs(self) -> List[str]:
        """List all run IDs in storage.

        Returns:
            List of run IDs
        """
        pass

    @abstractmethod
    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists in storage.

        Args:
            run_id: Unique identifier for the run

        Returns:
            True if run exists, False otherwise
        """
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """Delete all data for a run.

        Args:
            run_id: Unique identifier for the run
        """
        pass

    def close(self) -> None:
        """Close the storage backend and release resources.

        Optional method for cleanup. Called when storage is no longer needed.
        """
        pass

    # Canonical lifecycle semantics (vNext).
    def start_run(
        self,
        run_id: str,
        *,
        experiment_id: str = "default",
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Start a run and persist initial metadata."""
        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "config_snapshot": dict(config or {}),
        }
        self.save_run_metadata(run_id, metadata)

    def append_generation_record(
        self,
        run_id: str,
        record: GenerationRecord,
        *,
        cache_key: str | None = None,
    ) -> None:
        """Append a generation record to a run."""
        _ = cache_key  # cache keys are backend-specific
        self.save_generation_record(run_id, record)

    def append_evaluation_record(
        self,
        run_id: str,
        generation_record: GenerationRecord,
        evaluation_record: EvaluationRecord,
    ) -> None:
        """Append an evaluation record to a run."""
        self.save_evaluation_record(run_id, generation_record, evaluation_record)

    def complete_run(self, run_id: str) -> None:
        """Mark a run as completed."""
        metadata = self.load_run_metadata(run_id)
        metadata["status"] = "completed"
        self.save_run_metadata(run_id, metadata)

    def fail_run(self, run_id: str, error_message: str) -> None:
        """Mark a run as failed with an error message."""
        metadata = self.load_run_metadata(run_id)
        metadata["status"] = "failed"
        metadata["error_message"] = error_message
        self.save_run_metadata(run_id, metadata)


class LocalFileStorageBackend(StorageBackend):
    """StorageBackend adapter over ExperimentStorage."""

    def __init__(self, storage_path: str | Path):
        """Initialize with path to storage directory.

        Args:
            storage_path: Path to storage directory
        """
        from themis.storage import ExperimentStorage

        self._storage = ExperimentStorage(storage_path)

    @property
    def experiment_storage(self):
        """Expose underlying ExperimentStorage."""
        return self._storage

    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Save run metadata."""
        experiment_id = metadata.get("experiment_id", "default")
        if not self._storage.run_metadata_exists(run_id):
            self._storage.start_run(
                run_id,
                experiment_id=experiment_id,
                config=metadata,
            )
            return

        run_metadata = self._storage._load_run_metadata(run_id)
        run_metadata.experiment_id = experiment_id
        run_metadata.config_snapshot = dict(metadata)
        status = metadata.get("status")
        if status is not None:
            from themis.storage import RunStatus

            run_metadata.status = RunStatus(status)
        if "error_message" in metadata:
            run_metadata.error_message = metadata.get("error_message")
        run_metadata.updated_at = metadata.get("updated_at", run_metadata.updated_at)
        self._storage._save_run_metadata(run_metadata)

    def load_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load run metadata."""
        metadata = self._storage._load_run_metadata(run_id)
        payload = asdict(metadata)
        payload["status"] = metadata.status.value
        return payload

    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        """Save generation record."""
        self._ensure_run_exists(run_id)
        self._storage.append_record(run_id, record)

    def load_generation_records(self, run_id: str) -> List[GenerationRecord]:
        """Load generation records."""
        cached = self._storage.load_cached_records(run_id)
        return list(cached.values())

    def save_evaluation_record(
        self,
        run_id: str,
        generation_record: GenerationRecord,
        record: EvaluationRecord,
    ) -> None:
        """Save evaluation record."""
        self._ensure_run_exists(run_id)
        self._storage.append_evaluation(run_id, generation_record, record)

    def load_evaluation_records(self, run_id: str) -> Dict[str, EvaluationRecord]:
        """Load evaluation records."""
        return self._storage.load_cached_evaluations(run_id)

    def save_report(self, run_id: str, report: ExperimentReport) -> None:
        """Save report."""
        self._ensure_run_exists(run_id)
        report_path = self._storage.get_run_path(run_id) / "report.pkl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("wb") as handle:
            pickle.dump(report, handle)

        metadata_path = self._storage.get_run_path(run_id) / "report_metadata.json"
        metadata = {
            "run_id": run_id,
            "generation_results": len(report.generation_results),
            "failures": len(report.failures),
            "report_metadata": report.metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_report(self, run_id: str) -> ExperimentReport:
        """Load report."""
        report_path = self._storage.get_run_path(run_id) / "report.pkl"
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found for run {run_id}")
        with report_path.open("rb") as handle:
            report = pickle.load(handle)
        if not isinstance(report, ExperimentReport):
            raise TypeError(f"Invalid report payload for run {run_id}")
        return report

    def list_runs(self) -> List[str]:
        """List runs."""
        return [run.run_id for run in self._storage.list_runs()]

    def run_exists(self, run_id: str) -> bool:
        """Check if run exists."""
        return self._storage.run_metadata_exists(run_id)

    def delete_run(self, run_id: str) -> None:
        """Delete run."""
        self._storage.delete_run(run_id)

    def start_run(
        self,
        run_id: str,
        *,
        experiment_id: str = "default",
        config: Dict[str, Any] | None = None,
    ) -> None:
        self._storage.start_run(
            run_id, experiment_id=experiment_id, config=config or {}
        )

    def append_generation_record(
        self,
        run_id: str,
        record: GenerationRecord,
        *,
        cache_key: str | None = None,
    ) -> None:
        self._ensure_run_exists(run_id)
        self._storage.append_record(run_id, record, cache_key=cache_key)

    def append_evaluation_record(
        self,
        run_id: str,
        generation_record: GenerationRecord,
        evaluation_record: EvaluationRecord,
    ) -> None:
        self._ensure_run_exists(run_id)
        self._storage.append_evaluation(run_id, generation_record, evaluation_record)

    def complete_run(self, run_id: str) -> None:
        self._storage.complete_run(run_id)

    def fail_run(self, run_id: str, error_message: str) -> None:
        self._storage.fail_run(run_id, error_message)

    def _ensure_run_exists(self, run_id: str) -> None:
        if not self._storage.run_metadata_exists(run_id):
            self._storage.start_run(run_id, experiment_id="default")


__all__ = [
    "StorageBackend",
    "LocalFileStorageBackend",
]
