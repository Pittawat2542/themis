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

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from themis.core.entities import (
    EvaluationRecord,
    ExperimentReport,
    GenerationRecord,
)


class RecordStore(ABC):
    """Interface for managing generation and evaluation records."""

    @abstractmethod
    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        pass

    @abstractmethod
    def load_generation_records(self, run_id: str) -> List[GenerationRecord]:
        pass

    @abstractmethod
    def save_evaluation_record(
        self,
        run_id: str,
        generation_record: GenerationRecord,
        record: EvaluationRecord,
    ) -> None:
        pass

    @abstractmethod
    def load_evaluation_records(self, run_id: str) -> Dict[str, EvaluationRecord]:
        pass


class RunManager(ABC):
    """Interface for managing run lifecycle, metadata, and reports."""

    @abstractmethod
    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def load_run_metadata(self, run_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save_report(self, run_id: str, report: ExperimentReport) -> None:
        pass

    @abstractmethod
    def load_report(self, run_id: str) -> ExperimentReport:
        pass

    @abstractmethod
    def list_run_ids(self) -> List[str]:
        pass

    @abstractmethod
    def run_exists(self, run_id: str) -> bool:
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        pass


class StorageBackend(RecordStore, RunManager):
    """Combined interface for storage backends.

    Implement this interface to create custom storage solutions.
    All methods should be thread-safe if used with concurrent workers.
    """

    def close(self) -> None:
        """Close the storage backend and release resources."""
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
        *,
        evaluation_config: Dict[str, Any] | None = None,
    ) -> None:
        """Append an evaluation record to a run."""
        _ = evaluation_config
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

    def cache_dataset(self, run_id: str, dataset: List[Dict[str, Any]]) -> None:
        """Cache the dataset for resumability (optional for backends)."""
        pass

    def load_dataset(self, run_id: str) -> List[Dict[str, Any]] | None:
        """Load cached dataset (optional for backends)."""
        return None

    def load_cached_records(self, run_id: str) -> Dict[str, GenerationRecord]:
        """Load records mapped by cache_key for resumability (optional)."""
        return {}

    def load_cached_evaluations(
        self, run_id: str, evaluation_config: Dict[str, Any] | None = None
    ) -> Dict[str, EvaluationRecord]:
        """Load evaluations mapped by cache_key for resumability (optional)."""
        return {}

    def get_run_path(self, run_id: str) -> Any | None:
        """Get underlying storage path or identifier for this run."""
        return None


__all__ = [
    "StorageBackend",
    "RecordStore",
    "RunManager",
]
