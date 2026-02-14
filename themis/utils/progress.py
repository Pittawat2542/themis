"""Progress reporting utilities for Themis."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from themis.utils import logging_utils

if TYPE_CHECKING:
    from themis.experiment.integration_manager import IntegrationManager

logger = logging.getLogger(__name__)


class ProgressReporter(ABC):
    """Abstract base class for progress reporting."""

    @abstractmethod
    def __enter__(self) -> ProgressReporter:
        """Start the progress reporter."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the progress reporter."""
        pass

    @abstractmethod
    def add_task(self, description: str, total: int | None = None) -> int:
        """Add a new task to track.

        Returns:
            Task ID
        """
        pass

    @abstractmethod
    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        """Update task progress."""
        pass


class RichProgressReporter(ProgressReporter):
    """Progress reporter using Rich's dynamic progress bars."""

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(
                bar_width=40,
                style="dim white",
                complete_style="green",
                finished_style="bold green",
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•", style="dim"),
            MofNCompleteColumn(),
            TextColumn("•", style="dim"),
            TimeElapsedColumn(),
            TextColumn("•", style="dim"),
            TimeRemainingColumn(),
        )

    def __enter__(self) -> ProgressReporter:
        self._progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._progress.stop()

    def add_task(self, description: str, total: int | None = None) -> int:
        return self._progress.add_task(description, total=total)

    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        self._progress.update(task_id, advance=advance, **kwargs)


class LogProgressReporter(ProgressReporter):
    """Progress reporter using standard logging (JSON-friendly)."""

    def __init__(self) -> None:
        self._tasks: dict[int, dict[str, Any]] = {}
        self._next_id = 0
        self._last_log_time: dict[int, float] = {}
        self._lock = __import__("threading").Lock()

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Log completion for all tasks
        with self._lock:
            for task_id, task in self._tasks.items():
                self._log_progress(task_id, task)

    def add_task(self, description: str, total: int | None = None) -> int:
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            self._tasks[task_id] = {
                "description": description,
                "total": total,
                "completed": 0,
                "start_time": time.time(),
            }
            self._last_log_time[task_id] = time.time()

        logger.info(
            "%s (Total: %s)",
            description,
            total,
            extra={
                "event": "progress_start",
                "task_id": task_id,
                "description": description,
                "total": total,
            },
        )
        return task_id

    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return

            task["completed"] += advance
            task.update(kwargs)

            # Debounce logs: log every 10% or every 30 seconds
            total = task["total"]
            completed = task["completed"]

            should_log = False
            now = time.time()

            if total:
                # Check percentage
                percent = (completed / total) * 100
                last_percent = task.get("last_log_percent", 0)
                if percent - last_percent >= 10:
                    should_log = True
                    task["last_log_percent"] = percent
            else:
                # If total is unknown, log every 100 items, or every 10 if count is small
                if completed < 100 and completed % 10 == 0:
                    should_log = True
                elif completed % 100 == 0:
                    should_log = True

            if now - self._last_log_time.get(task_id, 0) > 10:  # Log every 10s at least
                should_log = True

            if total and completed == total:
                should_log = True

            if should_log:
                self._log_progress(task_id, task)
                self._last_log_time[task_id] = now

    def _log_progress(self, task_id: int, task: dict[str, Any]) -> None:
        total = task["total"]
        completed = task["completed"]
        description = task["description"]
        percent = (completed / total * 100) if total else 0

        extra = {
            "event": "progress_update",
            "task_id": task_id,
            "description": description,
            "completed": completed,
            "total": total,
            "percent": percent,
        }

        # Add any extra metrics passed in kwargs
        for k, v in task.items():
            if k not in [
                "description",
                "total",
                "completed",
                "start_time",
                "last_log_percent",
            ]:
                extra[k] = v

        logger.info(
            "%s: %s/%s (%.1f%%)",
            description,
            completed,
            total or "?",
            percent,
            extra=extra,
        )


class WandBProgressReporter(ProgressReporter):
    """Progress reporter that logs metrics to WandB."""

    def __init__(self, integration_manager: IntegrationManager) -> None:
        self._integrations = integration_manager
        self._tasks: dict[int, dict[str, Any]] = {}

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def add_task(self, description: str, total: int | None = None) -> int:
        # We don't track task IDs strictly here, just pass through updates
        # But we need to return a dummy ID if we don't have one
        return 0

    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        # Log progress metrics
        metrics = {}
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                metrics[f"progress/{k}"] = v

        if metrics:
            self._integrations.log_metrics(metrics)


class CompositeProgressReporter(ProgressReporter):
    """Progress reporter that delegates to multiple other reporters."""

    def __init__(self, reporters: list[ProgressReporter]) -> None:
        self._reporters = reporters

    def __enter__(self) -> ProgressReporter:
        for reporter in self._reporters:
            reporter.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for reporter in self._reporters:
            reporter.__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, description: str, total: int | None = None) -> int:
        # We assume all reporters return the same task ID or handle mapping internally
        # For simplicity, we return the ID from the first reporter
        first_id = 0
        for i, reporter in enumerate(self._reporters):
            tid = reporter.add_task(description, total)
            if i == 0:
                first_id = tid
        return first_id

    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        for reporter in self._reporters:
            reporter.update(task_id, advance, **kwargs)


def get_progress_reporter(
    integration_manager: IntegrationManager | None = None,
) -> ProgressReporter:
    """Get the appropriate progress reporter based on logging configuration.

    Args:
        integration_manager: Optional integration manager for WandB reporting
    """
    reporters: list[ProgressReporter] = []

    fmt = logging_utils.get_log_format()
    if fmt == "json":
        reporters.append(LogProgressReporter())
    else:
        reporters.append(RichProgressReporter())

    if integration_manager and integration_manager.has_wandb:
        reporters.append(WandBProgressReporter(integration_manager))

    if len(reporters) == 1:
        return reporters[0]

    return CompositeProgressReporter(reporters)
