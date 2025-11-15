"""Distributed tracing and observability utilities.

This module provides lightweight span-based tracing for understanding
experiment execution performance and behavior. Tracing is opt-in and
disabled by default for minimal performance impact.

Examples:
    # Enable tracing
    tracing.enable()

    # Use context manager for automatic span management
    with tracing.span("my_operation", task_id="123"):
        # ... do work ...
        with tracing.span("subprocess"):
            # ... nested work ...
            pass

    # Get trace for analysis
    trace = tracing.get_trace()
    print(f"Total time: {trace.duration_ms()}ms")

    # Export trace
    tracing.export_json("trace.json")
    tracing.disable()
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    """Represents a traced operation with timing and metadata.

    Spans can be nested to create a tree of operations showing
    how time is spent during experiment execution.
    """

    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent: Span | None = None
    children: list[Span] = field(default_factory=list)
    span_id: str = field(default_factory=lambda: str(time.time_ns()))

    def duration_ms(self) -> float:
        """Calculate span duration in milliseconds.

        Returns:
            Duration in milliseconds, or time elapsed so far if span not closed
        """
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def is_complete(self) -> bool:
        """Check if span has been closed."""
        return self.end_time is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms() if self.is_complete() else None,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }

    def find_spans(self, name: str) -> list[Span]:
        """Find all spans with given name in this span and descendants.

        Args:
            name: Span name to search for

        Returns:
            List of matching spans
        """
        matches = []
        if self.name == name:
            matches.append(self)

        for child in self.children:
            matches.extend(child.find_spans(name))

        return matches

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for this span tree.

        Returns:
            Dictionary with timing statistics by span name
        """
        summary: dict[str, dict[str, Any]] = {}

        def collect(span: Span):
            if span.name not in summary:
                summary[span.name] = {
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": 0.0,
                }

            stats = summary[span.name]
            if span.is_complete():
                duration = span.duration_ms()
                stats["count"] += 1
                stats["total_ms"] += duration
                stats["min_ms"] = min(stats["min_ms"], duration)
                stats["max_ms"] = max(stats["max_ms"], duration)

            for child in span.children:
                collect(child)

        collect(self)

        # Calculate averages
        for stats in summary.values():
            if stats["count"] > 0:
                stats["avg_ms"] = stats["total_ms"] / stats["count"]

        return summary


class TracingContext:
    """Thread-local tracing context.

    This class manages the current span stack for each thread, allowing
    nested spans and proper parent-child relationships.
    """

    def __init__(self):
        self._local = threading.local()
        self._enabled = False

    def enable(self) -> None:
        """Enable tracing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def _get_state(self):
        """Get thread-local state, initializing if needed."""
        if not hasattr(self._local, "root"):
            self._local.root = None
            self._local.current = None
        return self._local

    def get_root(self) -> Span | None:
        """Get the root span for current thread."""
        return self._get_state().root

    def get_current(self) -> Span | None:
        """Get the current active span for current thread."""
        return self._get_state().current

    @contextmanager
    def span(self, name: str, **metadata) -> Iterator[Span | None]:
        """Create a traced span as a context manager.

        Args:
            name: Name of the span
            **metadata: Additional metadata to attach to span

        Yields:
            Span object if tracing is enabled, None otherwise
        """
        if not self._enabled:
            yield None
            return

        state = self._get_state()
        span_obj = Span(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata,
            parent=state.current,
        )

        # Link to parent
        if state.current is not None:
            state.current.children.append(span_obj)
        else:
            # This is the root span
            state.root = span_obj

        # Make this the current span
        prev_current = state.current
        state.current = span_obj

        try:
            yield span_obj
        finally:
            # Close span
            span_obj.end_time = time.perf_counter()

            # Restore previous current
            state.current = prev_current

    def reset(self) -> None:
        """Reset tracing state for current thread."""
        state = self._get_state()
        state.root = None
        state.current = None


# Global tracing context
_global_context = TracingContext()


# Public API functions
def enable() -> None:
    """Enable tracing globally."""
    _global_context.enable()


def disable() -> None:
    """Disable tracing globally."""
    _global_context.disable()


def is_enabled() -> bool:
    """Check if tracing is enabled."""
    return _global_context.is_enabled()


@contextmanager
def span(name: str, **metadata) -> Iterator[Span | None]:
    """Create a traced span.

    This is the main API for creating spans. Use as a context manager:

        with tracing.span("my_operation", task_id="123"):
            # ... do work ...
            pass

    Args:
        name: Name of the span
        **metadata: Additional metadata to attach to span

    Yields:
        Span object if tracing is enabled, None otherwise
    """
    with _global_context.span(name, **metadata) as s:
        yield s


def get_trace() -> Span | None:
    """Get the root span for the current thread's trace.

    Returns:
        Root span, or None if no trace exists
    """
    return _global_context.get_root()


def reset() -> None:
    """Reset tracing state for current thread."""
    _global_context.reset()


def export_json(filepath: str, indent: int = 2) -> None:
    """Export current trace to JSON file.

    Args:
        filepath: Path to write JSON file
        indent: JSON indentation level (default: 2)
    """
    trace = get_trace()
    if trace is None:
        raise ValueError("No trace to export")

    with open(filepath, "w") as f:
        json.dump(trace.to_dict(), f, indent=indent)


def get_summary() -> dict[str, Any]:
    """Get summary statistics for current trace.

    Returns:
        Dictionary with timing statistics by span name

    Raises:
        ValueError: If no trace exists
    """
    trace = get_trace()
    if trace is None:
        raise ValueError("No trace to summarize")

    return trace.get_summary()


__all__ = [
    "Span",
    "TracingContext",
    "enable",
    "disable",
    "is_enabled",
    "span",
    "get_trace",
    "reset",
    "export_json",
    "get_summary",
]
