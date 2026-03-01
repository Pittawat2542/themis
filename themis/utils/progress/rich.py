"""Progress reporter using Rich's dynamic progress bars."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from themis.utils.progress.base import ProgressReporter


class RichProgressReporter(ProgressReporter):
    """Progress reporter using Rich's dynamic progress bars."""

    def __init__(self) -> None:
        # On Windows, Rich may fall back to a legacy renderer that uses the
        # current code page (typically CP1252) and cannot encode the braille
        # spinner characters.  Passing legacy_windows=False forces Rich to use
        # the VT-compatible path, which supports full Unicode on modern Windows
        # terminals and all CI runners.
        _console = Console(
            file=sys.stdout,
            legacy_windows=False,
            highlight=False,
        )
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
            console=_console,
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
