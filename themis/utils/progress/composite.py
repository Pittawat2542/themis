"""Progress reporter that delegates to multiple other reporters."""

from __future__ import annotations

from typing import Any

from themis.utils.progress.base import ProgressReporter


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
