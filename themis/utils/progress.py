"""Simple CLI-friendly progress reporter."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter(AbstractContextManager["ProgressReporter"]):
    def __init__(
        self,
        *,
        total: int | None,
        description: str = "Processing",
        unit: str = "sample",
        leave: bool = False,
    ) -> None:
        self._total = total
        self._description = description
        self._unit = unit
        self._leave = leave
        self._progress: Progress | None = None
        self._task_id = None

    def __enter__(self) -> "ProgressReporter":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def start(self) -> None:
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=not self._leave,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self._description, total=self._total
            )

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def increment(self, step: int = 1) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=step)

    def on_result(self, _record: Any) -> None:
        self.increment()

    def as_callback(self) -> Callable[[Any], None]:
        return self.on_result


__all__ = ["ProgressReporter"]
