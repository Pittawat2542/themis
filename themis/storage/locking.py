"""Concurrency control for storage operations."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path

# fcntl is Unix-only
if sys.platform == "win32":
    FCNTL_AVAILABLE = False
else:
    try:
        import fcntl

        FCNTL_AVAILABLE = True
    except ImportError:
        FCNTL_AVAILABLE = False

logger = logging.getLogger(__name__)


class LockManager:
    """Manages file locks for concurrent access."""

    def __init__(self) -> None:
        # run_id -> (fd, count) for reentrant locks
        self._locks: dict[str, tuple[int, int]] = {}
        # run_id -> thread_id of owner
        self._lock_owners: dict[str, int] = {}

    @contextlib.contextmanager
    def acquire(self, run_id: str, lock_path: Path):
        """Acquire exclusive lock for run directory with timeout (reentrant).

        Args:
            run_id: Unique run identifier
            lock_path: Path to lock file

        Yields:
            None
        """
        # Reentrant only for the same thread.
        owner = self._lock_owners.get(run_id)
        thread_id = threading.get_ident()
        if run_id in self._locks and owner == thread_id:
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
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Open lock file (OS-independent flags)
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)

        try:
            # Acquire exclusive lock with timeout
            self._acquire_os_lock(lock_fd, run_id, lock_path, timeout=30)

            self._locks[run_id] = (lock_fd, 1)
            self._lock_owners[run_id] = thread_id
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
        self, lock_fd: int, run_id: str, lock_path: Path, timeout: int = 30
    ) -> None:
        """Acquire OS-specific file lock with timeout."""
        if sys.platform == "win32":
            # Windows file locking with retry
            try:
                import msvcrt
            except ImportError:
                # msvcrt not available - single-process mode
                logger.debug("msvcrt not available. Single-process mode only.")
                return

            start_time = time.time()
            while True:
                try:
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    break  # Lock acquired
                except OSError as e:
                    # Lock is held by another thread/process
                    if time.time() - start_time > timeout:
                        try:
                            os.close(lock_fd)
                        except OSError:
                            pass
                        raise TimeoutError(
                            f"Failed to acquire lock for run {run_id} after {timeout}s on Windows. "
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
                        except OSError:
                            pass
                        raise TimeoutError(
                            f"Failed to acquire lock for run {run_id} after {timeout}s. "
                            f"Try: rm -f {lock_path}"
                        ) from e
                    time.sleep(0.1)  # Wait 100ms before retry
        else:
            # No locking available - single-process mode
            logger.debug(
                "File locking not available on this platform. "
                "Storage will work in single-process mode only."
            )

    def _release_os_lock(self, lock_fd: int, run_id: str) -> None:
        """Release OS-specific file lock."""
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
        self._lock_owners.pop(run_id, None)
