"""Sequential execution backend for debugging."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, TypeVar

from themis.backends.execution.base import ExecutionBackend

T = TypeVar("T")
R = TypeVar("R")


class SequentialExecutionBackend(ExecutionBackend):
    """Sequential execution backend for debugging.

    Executes tasks one at a time without parallelism.
    Useful for debugging, testing, or when parallelism causes issues.
    """

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        *,
        max_workers: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Iterator[R]:
        """Execute function sequentially.

        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            max_workers: Ignored (no parallelism)
            timeout: Timeout for each task (seconds)
            **kwargs: Ignored

        Yields:
            Results in input order
        """
        for item in items:
            result = func(item)
            yield result

    def shutdown(self) -> None:
        """No-op for sequential execution."""
        pass
