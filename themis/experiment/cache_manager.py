"""Cache management for experiment resumability and storage."""

from __future__ import annotations

from typing import Sequence

from themis.core.entities import EvaluationRecord, GenerationRecord
from themis.experiment import storage as experiment_storage


class CacheManager:
    """Manages experiment caching and resumability.

    This class handles all storage-related operations including:
    - Loading cached generation records
    - Loading cached evaluations
    - Saving datasets for resumability
    - Saving generation records and evaluations

    Single Responsibility: Cache and storage management
    """

    def __init__(
        self,
        storage: experiment_storage.ExperimentStorage | None,
        enable_resume: bool = True,
        enable_cache: bool = True,
    ) -> None:
        """Initialize cache manager.

        Args:
            storage: Storage backend (None disables caching)
            enable_resume: Whether to load cached results on resume
            enable_cache: Whether to save new results to cache
        """
        self._storage = storage
        self._enable_resume = enable_resume
        self._enable_cache = enable_cache

    @property
    def has_storage(self) -> bool:
        """Check if storage is available."""
        return self._storage is not None

    def cache_dataset(self, run_id: str, dataset: Sequence[dict[str, object]]) -> None:
        """Cache dataset for future resumability.

        Args:
            run_id: Unique run identifier
            dataset: Dataset samples to cache
        """
        if self._storage is not None and self._enable_cache:
            self._storage.cache_dataset(run_id, list(dataset))

    def load_cached_records(self, run_id: str) -> dict[str, GenerationRecord]:
        """Load cached generation records for resuming.

        Args:
            run_id: Unique run identifier

        Returns:
            Dictionary mapping cache keys to generation records
        """
        if not self._enable_resume or self._storage is None:
            return {}
        return self._storage.load_cached_records(run_id)

    def load_cached_evaluations(
        self, run_id: str, evaluation_config: dict | None = None
    ) -> dict[str, EvaluationRecord]:
        """Load cached evaluation records for resuming.

        Args:
            run_id: Unique run identifier
            evaluation_config: Evaluation configuration (metrics, extractor) for cache matching

        Returns:
            Dictionary mapping cache keys to evaluation records
        """
        if not self._enable_resume or self._storage is None:
            return {}
        return self._storage.load_cached_evaluations(run_id, evaluation_config=evaluation_config)

    def save_generation_record(
        self,
        run_id: str,
        record: GenerationRecord,
        cache_key: str,
    ) -> None:
        """Save a single generation record.

        Args:
            run_id: Unique run identifier
            record: Generation record to save
            cache_key: Cache key for this record
        """
        if self._storage is not None and self._enable_cache:
            self._storage.append_record(run_id, record, cache_key=cache_key)

    def save_evaluation_record(
        self,
        run_id: str,
        generation_record: GenerationRecord,
        evaluation_record: EvaluationRecord,
        evaluation_config: dict | None = None,
    ) -> None:
        """Save a single evaluation record.

        Args:
            run_id: Unique run identifier
            generation_record: Corresponding generation record
            evaluation_record: Evaluation record to save
            evaluation_config: Evaluation configuration for cache invalidation
        """
        if self._storage is not None and self._enable_cache:
            self._storage.append_evaluation(
                run_id, generation_record, evaluation_record, evaluation_config=evaluation_config
            )

    def get_run_path(self, run_id: str) -> str | None:
        """Get filesystem path for a run.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to run directory, or None if no storage
        """
        if self._storage is None:
            return None
        return str(self._storage.get_run_path(run_id))


__all__ = ["CacheManager"]
