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
    def save_evaluation_record(self, run_id: str, record: EvaluationRecord) -> None:
        """Save an evaluation record.
        
        Args:
            run_id: Unique identifier for the run
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


class LocalFileStorageBackend(StorageBackend):
    """Adapter for the existing ExperimentStorage implementation.
    
    This class wraps the current file-based storage implementation
    to conform to the StorageBackend interface.
    
    Note:
        This is a compatibility layer. New code should use the interface,
        but existing storage logic is preserved.
    """
    
    def __init__(self, storage_path: str | Path):
        """Initialize with path to storage directory.
        
        Args:
            storage_path: Path to storage directory
        """
        from themis.experiment.storage import ExperimentStorage
        self._storage = ExperimentStorage(storage_path)
    
    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Save run metadata."""
        experiment_id = metadata.get("experiment_id", "default")
        self._storage.start_run(run_id, experiment_id=experiment_id)
    
    def load_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load run metadata."""
        # Note: Current storage doesn't have a direct method for this
        # This is a limitation of the adapter pattern
        raise NotImplementedError("Use ExperimentStorage directly for now")
    
    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        """Save generation record."""
        self._storage.append_record(run_id, record)
    
    def load_generation_records(self, run_id: str) -> List[GenerationRecord]:
        """Load generation records."""
        cached = self._storage.load_cached_records(run_id)
        return list(cached.values())
    
    def save_evaluation_record(self, run_id: str, record: EvaluationRecord) -> None:
        """Save evaluation record."""
        self._storage.append_evaluation(run_id, record)
    
    def load_evaluation_records(self, run_id: str) -> Dict[str, EvaluationRecord]:
        """Load evaluation records."""
        return self._storage.load_cached_evaluations(run_id)
    
    def save_report(self, run_id: str, report: ExperimentReport) -> None:
        """Save report."""
        self._storage.save_report(run_id, report)
    
    def load_report(self, run_id: str) -> ExperimentReport:
        """Load report."""
        return self._storage.load_report(run_id)
    
    def list_runs(self) -> List[str]:
        """List runs."""
        return self._storage.list_runs()
    
    def run_exists(self, run_id: str) -> bool:
        """Check if run exists."""
        return run_id in self._storage.list_runs()
    
    def delete_run(self, run_id: str) -> None:
        """Delete run."""
        # Note: Current storage doesn't have delete functionality
        raise NotImplementedError("Delete not implemented in current storage")


__all__ = [
    "StorageBackend",
    "LocalFileStorageBackend",
]
