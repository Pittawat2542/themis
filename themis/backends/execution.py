"""Execution backend interface for custom execution strategies.

This module defines the abstract interface for execution backends, allowing
users to implement custom execution strategies (distributed, GPU-accelerated,
async, etc.) without modifying Themis core code.

Example implementations:
- RayExecutionBackend: Distributed execution with Ray
- DaskExecutionBackend: Distributed execution with Dask
- AsyncExecutionBackend: Async/await based execution
- GPUBatchExecutionBackend: Batched GPU execution for vLLM
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ExecutionBackend(ABC):
    """Abstract interface for execution backends.
    
    Implement this interface to create custom execution strategies.
    
    Example:
        >>> class RayExecutionBackend(ExecutionBackend):
        ...     def __init__(self, num_cpus: int = 4):
        ...         import ray
        ...         if not ray.is_initialized():
        ...             ray.init(num_cpus=num_cpus)
        ...
        ...     def map(self, func, items, max_workers=None):
        ...         import ray
        ...         # Convert to Ray remote function
        ...         remote_func = ray.remote(func)
        ...         # Submit all tasks
        ...         futures = [remote_func.remote(item) for item in items]
        ...         # Get results as they complete
        ...         for future in futures:
        ...             yield ray.get(future)
    """
    
    @abstractmethod
    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        *,
        max_workers: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Iterator[R]:
        """Execute function over items in parallel.
        
        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            max_workers: Maximum number of parallel workers
            timeout: Timeout for each execution (seconds)
            **kwargs: Additional backend-specific options
            
        Yields:
            Results as they complete
            
        Note:
            Results may be yielded in any order (not necessarily input order).
            Implementation should handle errors gracefully.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the execution backend and release resources.
        
        Called when execution is complete. Should cleanup workers,
        connections, and other resources.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


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
            **kwargs: Ignored (for interface compatibility)
            
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


__all__ = [
    "ExecutionBackend",
    "LocalExecutionBackend",
    "SequentialExecutionBackend",
]
