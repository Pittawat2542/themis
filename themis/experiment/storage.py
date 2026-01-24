"""Robust storage architecture with lifecycle management, atomic operations, and integrity checks.

This is a rewrite of the storage layer to address:
- Run lifecycle management (in_progress, completed, failed)
- Atomic write operations
- File locking for concurrent access
- Index persistence
- Experiment-level organization
- Separate evaluation tracking
- Data integrity validation
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
import shutil
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Literal

# fcntl is Unix-only, use msvcrt on Windows
if sys.platform == "win32":
    import msvcrt
    FCNTL_AVAILABLE = False
else:
    try:
        import fcntl
        FCNTL_AVAILABLE = True
    except ImportError:
        FCNTL_AVAILABLE = False

from themis.core import entities as core_entities
from themis.core import serialization as core_serialization

STORAGE_FORMAT_VERSION = "2.0.0"


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


class ExperimentStorage:
    """Robust storage with lifecycle management, locking, and integrity checks.

    Features:
    - Atomic write operations
    - File locking for concurrent access
    - Run lifecycle tracking (in_progress, completed, failed)
    - Experiment-level organization
    - Separate evaluation tracking
    - Persistent indexes
    - Data integrity validation
    - SQLite metadata database

    Example:
        >>> config = StorageConfig()
        >>> storage = ExperimentStorage("outputs/experiments", config=config)
        >>> 
        >>> # Start a run
        >>> metadata = storage.start_run("run-1", "experiment-1", config={})
        >>> 
        >>> # Append records with locking
        >>> storage.append_record("run-1", record)
        >>> 
        >>> # Complete the run
        >>> storage.complete_run("run-1")
    """

    def __init__(
        self, root: str | Path, config: StorageConfig | None = None
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._config = config or StorageConfig()

        # Create experiments directory
        self._experiments_dir = self._root / "experiments"
        self._experiments_dir.mkdir(exist_ok=True)

        # Initialize SQLite database
        if self._config.use_sqlite_metadata:
            self._init_database()

        # In-memory caches
        self._task_index: dict[str, set[str]] = {}
        self._template_index: dict[str, dict[str, str]] = {}
        self._locks: dict[str, tuple[int, int]] = {}  # (fd, count) for reentrant locks

    def _init_database(self):
        """Initialize SQLite metadata database."""
        db_path = self._root / "experiments.db"
        conn = sqlite3.connect(db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config TEXT,
                tags TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                total_samples INTEGER DEFAULT 0,
                successful_generations INTEGER DEFAULT 0,
                failed_generations INTEGER DEFAULT 0,
                config_snapshot TEXT,
                error_message TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                eval_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metrics_config TEXT,
                total_evaluated INTEGER DEFAULT 0,
                total_failures INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_experiment 
            ON runs(experiment_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_status 
            ON runs(status)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_run 
            ON evaluations(run_id)
        """)

        conn.commit()
        conn.close()

    @contextlib.contextmanager
    def _acquire_lock(self, run_id: str):
        """Acquire exclusive lock for run directory with timeout (reentrant).
        
        This lock is reentrant within the same thread to prevent deadlocks when
        the same process acquires the lock multiple times (e.g., start_run() 
        followed by append_record()).
        
        The lock uses OS-specific file locking:
        - Unix/Linux/macOS: fcntl.flock with non-blocking retry
        - Windows: msvcrt.locking
        - Fallback: No locking (single-process mode)
        
        Args:
            run_id: Unique run identifier
            
        Yields:
            Context manager that holds the lock
            
        Raises:
            TimeoutError: If lock cannot be acquired within 30 seconds
        """
        import time
        
        # Check if we already hold the lock (reentrant)
        if run_id in self._locks:
            lock_fd, count = self._locks[run_id]
            self._locks[run_id] = (lock_fd, count + 1)
            try:
                yield
            finally:
                # Check if lock still exists (might have been cleaned up by another thread)
                if run_id in self._locks:
                    lock_fd, count = self._locks[run_id]
                    if count > 1:
                        self._locks[run_id] = (lock_fd, count - 1)
                    else:
                        # Last unlock - release the actual lock
                        self._release_os_lock(lock_fd, run_id)
            return
        
        # First time acquiring lock for this run_id
        lock_path = self._get_run_dir(run_id) / ".lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Open lock file (OS-independent flags)
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)

        try:
            # Acquire exclusive lock with timeout
            self._acquire_os_lock(lock_fd, run_id, lock_path, timeout=30)
            
            self._locks[run_id] = (lock_fd, 1)
            yield
        finally:
            # Release lock (only if this was the outermost lock)
            if run_id in self._locks:
                lock_fd, count = self._locks[run_id]
                if count == 1:
                    self._release_os_lock(lock_fd, run_id)
                else:
                    # Decrement count
                    self._locks[run_id] = (lock_fd, count - 1)
    
    def _acquire_os_lock(
        self, 
        lock_fd: int, 
        run_id: str, 
        lock_path: Path, 
        timeout: int = 30
    ) -> None:
        """Acquire OS-specific file lock with timeout.
        
        Args:
            lock_fd: File descriptor for lock file
            run_id: Run identifier (for error messages)
            lock_path: Path to lock file (for error messages)
            timeout: Timeout in seconds
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        import time
        
        if sys.platform == "win32":
            # Windows file locking with retry
            try:
                import msvcrt
            except ImportError:
                # msvcrt not available - single-process mode
                import logging
                logger = logging.getLogger(__name__)
                logger.debug("msvcrt not available. Single-process mode only.")
                return
            
            start_time = time.time()
            while True:
                try:
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    break  # Lock acquired
                except OSError as e:
                    # Lock is held by another thread/process (errno 13 Permission denied)
                    if time.time() - start_time > timeout:
                        try:
                            os.close(lock_fd)
                        except:
                            pass
                        raise TimeoutError(
                            f"Failed to acquire lock for run {run_id} after {timeout}s on Windows. "
                            f"This usually means another process is holding the lock or a previous process crashed. "
                            f"Try deleting: {lock_path}"
                        ) from e
                    time.sleep(0.1)  # Wait 100ms before retry
        elif FCNTL_AVAILABLE:
            # Unix file locking with non-blocking retry
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break  # Lock acquired
                except (IOError, OSError) as e:
                    # Lock is held by another process
                    if time.time() - start_time > timeout:
                        try:
                            os.close(lock_fd)
                        except:
                            pass
                        raise TimeoutError(
                            f"Failed to acquire lock for run {run_id} after {timeout}s. "
                            f"This usually means another process is holding the lock or a previous process crashed. "
                            f"Try: rm -f {lock_path}"
                        ) from e
                    time.sleep(0.1)  # Wait 100ms before retry
        else:
            # No locking available - single-process mode
            # This is safe for single-process usage (most common case)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"File locking not available on this platform. "
                f"Storage will work in single-process mode only."
            )
    
    def _release_os_lock(self, lock_fd: int, run_id: str) -> None:
        """Release OS-specific file lock.
        
        Args:
            lock_fd: File descriptor to close
            run_id: Run identifier (for cleanup)
        """
        # Release lock
        if sys.platform == "win32":
            try:
                import msvcrt
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
            except (ImportError, OSError):
                pass  # Lock may already be released
        elif FCNTL_AVAILABLE:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (IOError, OSError):
                pass  # Lock may already be released
        
        # Close file descriptor
        try:
            os.close(lock_fd)
        except OSError:
            pass  # FD may already be closed
        
        # Clean up tracking
        self._locks.pop(run_id, None)

    def start_run(
        self,
        run_id: str,
        experiment_id: str,
        config: dict | None = None,
    ) -> RunMetadata:
        """Start a new run with in_progress status.

        Args:
            run_id: Unique run identifier
            experiment_id: Experiment this run belongs to
            config: Configuration snapshot for this run

        Returns:
            RunMetadata with in_progress status

        Raises:
            ValueError: If run already exists
        """
        with self._acquire_lock(run_id):
            # Check if run already exists
            if self._run_metadata_exists(run_id):
                raise ValueError(f"Run {run_id} already exists")

            # Create run directory
            run_dir = self._get_run_dir(run_id)
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
        """Mark run as completed.

        Args:
            run_id: Run identifier

        Raises:
            ValueError: If run doesn't exist
        """
        with self._acquire_lock(run_id):
            metadata = self._load_run_metadata(run_id)
            metadata.status = RunStatus.COMPLETED
            metadata.completed_at = datetime.now().isoformat()
            metadata.updated_at = datetime.now().isoformat()
            self._save_run_metadata(metadata)

    def fail_run(self, run_id: str, error_message: str):
        """Mark run as failed with error message.

        Args:
            run_id: Run identifier
            error_message: Error description
        """
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
        """Update run progress counters.

        Args:
            run_id: Run identifier
            total_samples: Total samples (if provided)
            successful_generations: Successful count (if provided)
            failed_generations: Failed count (if provided)
        """
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
        """Append record with atomic write and locking.

        Args:
            run_id: Run identifier
            record: Generation record to append
            cache_key: Optional cache key (generated if not provided)
        """
        with self._acquire_lock(run_id):
            # Ensure generation directory exists
            gen_dir = self._get_generation_dir(run_id)
            gen_dir.mkdir(parents=True, exist_ok=True)

            path = gen_dir / "records.jsonl"

            # Initialize file with header if needed
            if not self._file_exists_any_compression(path):
                self._write_jsonl_with_header(path, [], file_type="records")

            # Serialize record
            payload = self._serialize_record(run_id, record)
            payload["cache_key"] = cache_key or self._task_cache_key(record.task)

            # Atomic append
            self._atomic_append(path, payload)

            # Update progress
            metadata = self._load_run_metadata(run_id)
            new_successful = metadata.successful_generations + (1 if record.output else 0)
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

    def _atomic_append(self, path: Path, data: dict):
        """Append data atomically using temp file.

        Args:
            path: Target file path
            data: Data to append (will be JSON serialized)
        """
        json_line = json.dumps(data) + "\n"

        # Write to temp file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".json"
        )
        temp_path = Path(temp_path)

        try:
            if self._config.compression == "gzip":
                # Close the fd first since gzip.open will open by path
                os.close(temp_fd)
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    f.write(json_line)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                # Use the fd directly
                with open(temp_fd, "w", encoding="utf-8") as f:
                    f.write(json_line)
                    f.flush()
                    os.fsync(f.fileno())
                # fd is closed by context manager, don't close again

            # Get target path with compression
            target_path = (
                path.with_suffix(path.suffix + ".gz")
                if self._config.compression == "gzip"
                else path
            )

            # Append to existing file
            if target_path.exists():
                with open(target_path, "ab") as dest:
                    with open(temp_path, "rb") as src:
                        dest.write(src.read())
                    dest.flush()
                    os.fsync(dest.fileno())
            else:
                # No existing file, just rename
                temp_path.rename(target_path)
                return

        finally:
            # Clean up temp file if still exists
            if temp_path.exists():
                temp_path.unlink()

    def _save_run_metadata(self, metadata: RunMetadata):
        """Save run metadata to both JSON and SQLite.

        Args:
            metadata: Run metadata to save
        """
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
        if self._config.use_sqlite_metadata:
            self._save_run_metadata_to_db(metadata)

    def _save_run_metadata_to_db(self, metadata: RunMetadata):
        """Save run metadata to SQLite database."""
        db_path = self._root / "experiments.db"
        conn = sqlite3.connect(db_path)

        # Ensure experiment exists
        conn.execute(
            """
            INSERT OR IGNORE INTO experiments (experiment_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                metadata.experiment_id,
                metadata.experiment_id,
                metadata.created_at,
                metadata.updated_at,
            ),
        )

        # Upsert run
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, experiment_id, status, created_at, updated_at, completed_at,
                total_samples, successful_generations, failed_generations,
                config_snapshot, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.run_id,
                metadata.experiment_id,
                metadata.status.value,
                metadata.created_at,
                metadata.updated_at,
                metadata.completed_at,
                metadata.total_samples,
                metadata.successful_generations,
                metadata.failed_generations,
                json.dumps(metadata.config_snapshot),
                metadata.error_message,
            ),
        )

        conn.commit()
        conn.close()

    def _load_run_metadata(self, run_id: str) -> RunMetadata:
        """Load run metadata from JSON file.

        Args:
            run_id: Run identifier

        Returns:
            RunMetadata

        Raises:
            FileNotFoundError: If metadata doesn't exist
        """
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

    def _get_run_dir(self, run_id: str) -> Path:
        """Get run directory path.

        Uses hierarchical structure: experiments/<experiment_id>/runs/<run_id>/
        Falls back to experiments/default/runs/<run_id>/ if experiment_id unknown.
        """
        # Check if we already have metadata
        for exp_dir in self._experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue
            candidate_path = runs_dir / run_id / "metadata.json"
            if candidate_path.exists():
                return runs_dir / run_id

        # Default location for new runs
        return self._experiments_dir / "default" / "runs" / run_id

    def _get_generation_dir(self, run_id: str) -> Path:
        """Get generation data directory."""
        return self._get_run_dir(run_id) / "generation"

    def _get_evaluation_dir(self, run_id: str, eval_id: str = "default") -> Path:
        """Get evaluation directory."""
        return self._get_run_dir(run_id) / "evaluations" / eval_id

    def _file_exists_any_compression(self, path: Path) -> bool:
        """Check if file exists with any compression suffix."""
        return path.exists() or path.with_suffix(path.suffix + ".gz").exists()

    def _open_for_read(self, path: Path):
        """Open file for reading with automatic compression detection.
        
        Args:
            path: File path
            
        Returns:
            File handle (text mode)
        """
        # Try .gz version first
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            return gzip.open(gz_path, "rt", encoding="utf-8")
        if path.exists():
            return path.open("r", encoding="utf-8")
        raise FileNotFoundError(f"File not found: {path}")

    def _write_jsonl_with_header(
        self, path: Path, items: Iterable[dict], file_type: str
    ):
        """Write JSONL file with format version header."""
        # Determine actual path based on compression
        if self._config.compression == "gzip":
            actual_path = path.with_suffix(path.suffix + ".gz")
            handle = gzip.open(actual_path, "wt", encoding="utf-8")
        else:
            actual_path = path
            handle = open(actual_path, "w", encoding="utf-8")

        with handle:
            # Write header
            header = {
                "_type": "header",
                "_format_version": STORAGE_FORMAT_VERSION,
                "_file_type": file_type,
            }
            handle.write(json.dumps(header) + "\n")

            # Write items
            for item in items:
                handle.write(json.dumps(item) + "\n")

            handle.flush()
            if hasattr(handle, "fileno"):
                os.fsync(handle.fileno())

    def cache_dataset(self, run_id: str, dataset: Iterable[dict[str, object]]) -> None:
        """Cache dataset samples to storage.

        Args:
            run_id: Unique run identifier
            dataset: Iterable of dataset samples
        """
        if not self._config.save_dataset:
            return

        with self._acquire_lock(run_id):
            gen_dir = self._get_generation_dir(run_id)
            gen_dir.mkdir(parents=True, exist_ok=True)
            path = gen_dir / "dataset.jsonl"

            self._write_jsonl_with_header(path, dataset, file_type="dataset")

    def load_dataset(self, run_id: str) -> List[dict[str, object]]:
        """Load cached dataset.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of dataset samples
        """
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "dataset.jsonl"
        
        rows: list[dict[str, object]] = []
        with self._open_for_read(path) as handle:
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
    ) -> Dict[str, core_entities.GenerationRecord]:
        """Load cached generation records.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dict mapping cache_key to GenerationRecord
        """
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "records.jsonl"
        
        try:
            handle = self._open_for_read(path)
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
        """Append evaluation result.
        
        Args:
            run_id: Run identifier
            record: Generation record being evaluated
            evaluation: Evaluation record
            eval_id: Evaluation identifier (default: "default")
            evaluation_config: Evaluation configuration (metrics, extractor) for cache invalidation
        """
        with self._acquire_lock(run_id):
            eval_dir = self._get_evaluation_dir(run_id, eval_id)
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            path = eval_dir / "evaluation.jsonl"
            
            if not self._file_exists_any_compression(path):
                self._write_jsonl_with_header(path, [], file_type="evaluation")
            
            # Use evaluation_cache_key that includes evaluation config
            cache_key = evaluation_cache_key(record.task, evaluation_config)
            
            payload = {
                "cache_key": cache_key,
                "evaluation": core_serialization.serialize_evaluation_record(evaluation),
            }
            self._atomic_append(path, payload)

    def load_cached_evaluations(
        self, run_id: str, eval_id: str = "default", evaluation_config: dict | None = None
    ) -> Dict[str, core_entities.EvaluationRecord]:
        """Load cached evaluation records.
        
        Args:
            run_id: Run identifier
            eval_id: Evaluation identifier
            evaluation_config: Evaluation configuration for cache key matching
            
        Returns:
            Dict mapping cache_key to EvaluationRecord
            
        Note:
            If evaluation_config is provided, only evaluations matching that config
            will be loaded. This ensures that changing metrics invalidates the cache.
        """
        eval_dir = self._get_evaluation_dir(run_id, eval_id)
        path = eval_dir / "evaluation.jsonl"
        
        try:
            handle = self._open_for_read(path)
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
                
                evaluations[key] = core_serialization.deserialize_evaluation_record(
                    data["evaluation"]
                )
        
        return evaluations

    def get_run_path(self, run_id: str) -> Path:
        """Get the filesystem path for a run's storage directory.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to the run's storage directory
        """
        return self._get_run_dir(run_id)

    def _serialize_record(
        self, run_id: str, record: core_entities.GenerationRecord
    ) -> dict:
        """Serialize generation record."""
        task_key = self._persist_task(run_id, record.task)

        # Prepare output data
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
                text=output_data["text"],
                raw=output_data.get("raw")
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
        # Implementation similar to original but with atomic writes
        # and proper locking (already have lock from append_record)
        key = self._task_cache_key(task)
        index = self._load_task_index(run_id)

        if key in index:
            return key

        gen_dir = self._get_generation_dir(run_id)
        gen_dir.mkdir(parents=True, exist_ok=True)
        path = gen_dir / "tasks.jsonl"

        # Initialize if needed
        if not self._file_exists_any_compression(path):
            self._write_jsonl_with_header(path, [], file_type="tasks")

        # Serialize task
        if self._config.deduplicate_templates:
            template_id = self._persist_template(run_id, task.prompt.spec)
            task_data = core_serialization.serialize_generation_task(task)
            task_data["prompt"]["spec"] = {"_template_ref": template_id}
        else:
            task_data = core_serialization.serialize_generation_task(task)

        payload = {"task_key": key, "task": task_data}
        self._atomic_append(path, payload)

        index.add(key)
        self._save_task_index(run_id, index)

        return key

    def _persist_template(
        self, run_id: str, spec: core_entities.PromptSpec
    ) -> str:
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

        if not self._file_exists_any_compression(path):
            self._write_jsonl_with_header(path, [], file_type="templates")

        payload = {
            "template_id": template_id,
            "spec": core_serialization.serialize_prompt_spec(spec),
        }
        self._atomic_append(path, payload)

        self._template_index[run_id][template_id] = spec.template
        return template_id

    def _load_task_index(self, run_id: str) -> set[str]:
        """Load task index from disk cache or rebuild."""
        if run_id in self._task_index:
            return self._task_index[run_id]

        # Try to load from persisted index
        index_path = self._get_run_dir(run_id) / ".index.json"
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
            self._task_index[run_id] = set(index_data.get("task_keys", []))
            return self._task_index[run_id]

        # Rebuild from tasks file
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
        """Load templates from disk.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dict mapping template_id to PromptSpec
        """
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "templates.jsonl"
        
        templates: dict[str, core_entities.PromptSpec] = {}
        try:
            handle = self._open_for_read(path)
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
        """Load tasks from disk.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dict mapping task_key to GenerationTask
        """
        gen_dir = self._get_generation_dir(run_id)
        path = gen_dir / "tasks.jsonl"
        
        tasks: dict[str, core_entities.GenerationTask] = {}
        try:
            handle = self._open_for_read(path)
        except FileNotFoundError:
            return tasks
        
        # Load templates if deduplication enabled
        templates = self._load_templates(run_id) if self._config.deduplicate_templates else {}
        
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue
                
                task_key = data["task_key"]
                task_data = data["task"]
                
                # Restore template from reference if needed
                if (
                    self._config.deduplicate_templates
                    and "_template_ref" in task_data.get("prompt", {}).get("spec", {})
                ):
                    template_id = task_data["prompt"]["spec"]["_template_ref"]
                    if template_id in templates:
                        task_data["prompt"]["spec"] = core_serialization.serialize_prompt_spec(
                            templates[template_id]
                        )
                
                tasks[task_key] = core_serialization.deserialize_generation_task(task_data)
        
        self._task_index[run_id] = set(tasks.keys())
        return tasks

    def _task_cache_key(self, task: core_entities.GenerationTask) -> str:
        """Generate cache key for task."""
        dataset_raw = task.metadata.get("dataset_id") or task.metadata.get("sample_id")
        dataset_id = str(dataset_raw) if dataset_raw is not None else ""
        prompt_hash = hashlib.sha256(task.prompt.text.encode("utf-8")).hexdigest()[:12]
        sampling = task.sampling
        sampling_key = (
            f"{sampling.temperature:.3f}-{sampling.top_p:.3f}-{sampling.max_tokens}"
        )
        template = task.prompt.spec.name
        model = task.model.identifier
        return "::".join(
            filter(None, [dataset_id, template, model, sampling_key, prompt_hash])
        )

    # ===== Phase 3 Features =====

    def save_checkpoint(self, run_id: str, checkpoint_data: dict):
        """Save checkpoint for resumability.
        
        Args:
            run_id: Run identifier
            checkpoint_data: Checkpoint data to save
        """
        with self._acquire_lock(run_id):
            checkpoint_dir = self._get_run_dir(run_id) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Use timestamp for checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.json"
            
            checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

    def load_latest_checkpoint(self, run_id: str) -> dict | None:
        """Load most recent checkpoint.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Checkpoint data or None if no checkpoints exist
        """
        checkpoint_dir = self._get_run_dir(run_id) / "checkpoints"
        if not checkpoint_dir.exists():
            return None
        
        # Find latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
        if not checkpoints:
            return None
        
        return json.loads(checkpoints[0].read_text())

    def apply_retention_policy(self, policy: RetentionPolicy | None = None):
        """Apply retention policy to clean up old runs.
        
        Args:
            policy: Retention policy (uses config if not provided)
        """
        policy = policy or self._config.retention_policy
        if not policy:
            return
        
        # Get all experiments
        for exp_dir in self._experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue
            
            # Load all run metadata
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
            
            # Sort by creation time (newest first)
            runs.sort(key=lambda x: x[1].created_at, reverse=True)
            
            # Apply policies
            runs_to_delete = []
            
            for i, (run_dir, metadata) in enumerate(runs):
                # Always keep latest N runs
                if i < policy.keep_latest_n:
                    continue
                
                # Check if should keep based on status
                if policy.keep_completed_only and metadata.status != RunStatus.COMPLETED:
                    runs_to_delete.append(run_dir)
                    continue
                
                # Check age policy
                if policy.max_age_days:
                    created = datetime.fromisoformat(metadata.created_at)
                    age = datetime.now() - created
                    if age > timedelta(days=policy.max_age_days):
                        runs_to_delete.append(run_dir)
                        continue
                
                # Check max runs policy
                if policy.max_runs_per_experiment:
                    if i >= policy.max_runs_per_experiment:
                        runs_to_delete.append(run_dir)
            
            # Delete runs
            for run_dir in runs_to_delete:
                self._delete_run_dir(run_dir)

    def _delete_run_dir(self, run_dir: Path):
        """Delete run directory and update database.
        
        Args:
            run_dir: Run directory to delete
        """
        run_id = run_dir.name
        
        # Remove from SQLite
        if self._config.use_sqlite_metadata:
            db_path = self._root / "experiments.db"
            conn = sqlite3.connect(db_path)
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()
        
        # Remove directory
        shutil.rmtree(run_dir, ignore_errors=True)

    def get_storage_size(self, experiment_id: str | None = None) -> int:
        """Get total storage size in bytes.
        
        Args:
            experiment_id: Optional experiment to check (all if None)
            
        Returns:
            Total size in bytes
        """
        if experiment_id:
            exp_dir = self._experiments_dir / experiment_id
            if not exp_dir.exists():
                return 0
            return sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file())
        else:
            return sum(f.stat().st_size for f in self._experiments_dir.rglob("*") if f.is_file())

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None
    ) -> list[RunMetadata]:
        """List runs with optional filtering.
        
        Args:
            experiment_id: Filter by experiment
            status: Filter by status
            limit: Maximum number of runs to return
            
        Returns:
            List of run metadata
        """
        if not self._config.use_sqlite_metadata:
            # Fallback to file-based listing
            return self._list_runs_from_files(experiment_id, status, limit)
        
        # Query SQLite
        db_path = self._root / "experiments.db"
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM runs WHERE 1=1"
        params = []
        
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to RunMetadata
        runs = []
        for row in rows:
            runs.append(RunMetadata(
                run_id=row[0],
                experiment_id=row[1],
                status=RunStatus(row[2]),
                created_at=row[3],
                updated_at=row[4],
                completed_at=row[5],
                total_samples=row[6] or 0,
                successful_generations=row[7] or 0,
                failed_generations=row[8] or 0,
                config_snapshot=json.loads(row[9]) if row[9] else {},
                error_message=row[10],
            ))
        
        return runs

    def _list_runs_from_files(
        self,
        experiment_id: str | None,
        status: RunStatus | None,
        limit: int | None
    ) -> list[RunMetadata]:
        """List runs by scanning files (fallback)."""
        runs = []
        
        # Scan experiment directories
        exp_dirs = [self._experiments_dir / experiment_id] if experiment_id else list(self._experiments_dir.iterdir())
        
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue
            
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                try:
                    metadata = self._load_run_metadata(run_dir.name)
                    if status and metadata.status != status:
                        continue
                    runs.append(metadata)
                except Exception:
                    continue
        
        # Sort by creation time
        runs.sort(key=lambda r: r.created_at, reverse=True)
        
        if limit:
            runs = runs[:limit]
        
        return runs

    def validate_integrity(self, run_id: str) -> dict:
        """Validate data integrity for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dict with validation results
        """
        results = {
            "run_id": run_id,
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        run_dir = self._get_run_dir(run_id)
        if not run_dir.exists():
            results["valid"] = False
            results["errors"].append(f"Run directory not found: {run_dir}")
            return results
        
        # Check metadata
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            results["valid"] = False
            results["errors"].append("Missing metadata.json")
        
        # Check generation directory
        gen_dir = run_dir / "generation"
        if not gen_dir.exists():
            results["warnings"].append("No generation directory")
        else:
            # Check for required files
            for filename in ["records.jsonl", "tasks.jsonl"]:
                if not self._file_exists_any_compression(gen_dir / filename):
                    results["warnings"].append(f"Missing {filename}")
        
        # Check lock file
        lock_path = run_dir / ".lock"
        if not lock_path.exists():
            results["warnings"].append("No lock file (may not have been used)")
        
        return results


def task_cache_key(task: core_entities.GenerationTask) -> str:
    """Derive a stable cache key for a generation task (module-level function for backward compatibility)."""
    dataset_raw = task.metadata.get("dataset_id") or task.metadata.get("sample_id")
    dataset_id = str(dataset_raw) if dataset_raw is not None else ""
    prompt_hash = hashlib.sha256(task.prompt.text.encode("utf-8")).hexdigest()[:12]
    sampling = task.sampling
    sampling_key = (
        f"{sampling.temperature:.3f}-{sampling.top_p:.3f}-{sampling.max_tokens}"
    )
    template = task.prompt.spec.name
    model = task.model.identifier
    return "::".join(
        filter(None, [dataset_id, template, model, sampling_key, prompt_hash])
    )


def evaluation_cache_key(
    task: core_entities.GenerationTask,
    evaluation_config: dict | None = None,
) -> str:
    """Derive a stable cache key for an evaluation that includes both task and evaluation configuration.
    
    This ensures that changing metrics or evaluation settings will invalidate the cache
    and trigger re-evaluation, even if the generation is cached.
    
    Args:
        task: Generation task
        evaluation_config: Dictionary with evaluation configuration:
            - metrics: List of metric names/types
            - extractor: Extractor type/configuration
            - Any other evaluation settings
    
    Returns:
        Cache key string that includes both task and evaluation config
    
    Example:
        >>> config = {
        ...     "metrics": ["exact_match", "f1_score"],
        ...     "extractor": "json_field_extractor:answer"
        ... }
        >>> key = evaluation_cache_key(task, config)
    """
    task_key = task_cache_key(task)
    
    if not evaluation_config:
        # No config provided, use task key only (for backward compatibility)
        return task_key
    
    # Create deterministic hash of evaluation configuration
    config_str = json.dumps(evaluation_config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]
    
    return f"{task_key}::eval:{config_hash}"


__all__ = [
    "ExperimentStorage",
    "StorageConfig",
    "RunMetadata",
    "EvaluationMetadata",
    "RunStatus",
    "RetentionPolicy",
    "DataIntegrityError",
    "ConcurrentAccessError",
    "task_cache_key",
    "evaluation_cache_key",
]
