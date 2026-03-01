"""Local multi-threaded execution using ThreadPoolExecutor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, TypeVar

from themis.backends.execution.base import ExecutionBackend

T = TypeVar("T")
R = TypeVar("R")


class LocalExecutionBackend(ExecutionBackend):
    """Local multi-threaded execution using ThreadPoolExecutor.

    This is the default execution backend, using Python's built-in
    ThreadPoolExecutor for parallel execution.

    Attributes:
        executor: ThreadPoolExecutor instance
    """

    def __init__(self, max_workers: int = 4):
        """Initialize with number of workers.

        Args:
            max_workers: Maximum number of worker threads
        """
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        *,
        max_workers: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Iterator[R]:
        """Execute function over items using ThreadPoolExecutor.

        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            max_workers: Override default max_workers
            timeout: Timeout for each task (seconds)
            **kwargs: Backend-specific extras (unused by local backend)

        Yields:
            Results as they complete
        """
        workers = max_workers or self._max_workers

        # Create executor if not exists
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=workers)

        # Submit all tasks
        items_list = list(items)  # Materialize iterator
        futures = [self._executor.submit(func, item) for item in items_list]

        # Yield results as they complete
        for future in as_completed(futures, timeout=timeout):
            result = future.result()
            yield result

    def shutdown(self) -> None:
        """Shutdown the executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
