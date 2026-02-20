"""Storage models and data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class RunStatus(str, Enum):
    """Status of a run."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetentionPolicy:
    """Retention policy for automatic cleanup.

    Attributes:
        max_runs_per_experiment: Maximum runs to keep per experiment
        max_age_days: Maximum age in days for runs
        max_storage_gb: Maximum total storage in GB
        keep_completed_only: Only keep completed runs
        keep_latest_n: Always keep N most recent runs
    """

    max_runs_per_experiment: int | None = None
    max_age_days: int | None = None
    max_storage_gb: float | None = None
    keep_completed_only: bool = True
    keep_latest_n: int = 5


@dataclass
class StorageConfig:
    """Configuration for experiment storage behavior.

    Attributes:
        save_raw_responses: Save full API responses (default: False)
        save_dataset: Save dataset copy (default: True)
        compression: Compression format - "gzip" | "none" (default: "gzip")
        deduplicate_templates: Store templates once (default: True)
        enable_checksums: Add integrity checksums (default: True)
        use_sqlite_metadata: Use SQLite for metadata (default: True)
        checkpoint_interval: Save checkpoint every N records (default: 100)
        retention_policy: Automatic cleanup policy (default: None)
    """

    save_raw_responses: bool = False
    save_dataset: bool = True
    compression: Literal["none", "gzip"] = "gzip"
    deduplicate_templates: bool = True
    enable_checksums: bool = True
    use_sqlite_metadata: bool = True
    checkpoint_interval: int = 100
    retention_policy: RetentionPolicy | None = None


@dataclass
class RunMetadata:
    """Metadata for a run."""

    run_id: str
    experiment_id: str
    status: RunStatus
    created_at: str
    updated_at: str
    completed_at: str | None = None
    total_samples: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    config_snapshot: dict = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class EvaluationMetadata:
    """Metadata for an evaluation run."""

    eval_id: str
    run_id: str
    eval_name: str
    created_at: str
    metrics_config: dict = field(default_factory=dict)
    total_evaluated: int = 0
    total_failures: int = 0


class DataIntegrityError(Exception):
    """Raised when data integrity check fails."""

    pass


class ConcurrentAccessError(Exception):
    """Raised when concurrent access conflict detected."""

    pass
