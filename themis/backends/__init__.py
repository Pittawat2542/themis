"""Backend interfaces for extending Themis.

This module provides abstract interfaces for implementing custom backends:
- StorageBackend: Custom storage implementations (cloud, databases, etc.)
- ExecutionBackend: Custom execution strategies (distributed, async, etc.)

These interfaces allow advanced users to extend Themis without modifying core code.
"""

from themis.backends.execution import ExecutionBackend, LocalExecutionBackend
from themis.backends.storage import StorageBackend

__all__ = [
    "StorageBackend",
    "ExecutionBackend",
    "LocalExecutionBackend",
]
