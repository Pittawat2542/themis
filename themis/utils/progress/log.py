"""Progress reporter using standard logging (JSON-friendly)."""

from __future__ import annotations

import logging
import time
from typing import Any

from themis.utils.progress.base import ProgressReporter

logger = logging.getLogger(__name__)


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
