from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

from themis.progress.bus import ProgressEvent
from themis.progress.models import ProgressVerbosity, RunProgressSnapshot
from themis.types.enums import RunStage

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.live import Live

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProgressLogRenderer:
    """Logs progress snapshots using stdlib logging."""

    verbosity: ProgressVerbosity = ProgressVerbosity.NORMAL

    def __call__(self, event: ProgressEvent) -> None:
        snapshot = event.snapshot
        message = (
            "run=%s event=%s active_stage=%s processed=%s remaining=%s in_flight=%s"
        )
        args = (
            snapshot.run_id,
            event.event_type,
            snapshot.active_stage.value if snapshot.active_stage is not None else "-",
            snapshot.processed_items,
            snapshot.remaining_items,
            snapshot.in_flight_items,
        )
        if self.verbosity == ProgressVerbosity.DEBUG:
            logger.info(message, *args)
            for stage in RunStage:
                counts = snapshot.stage_counts.get(stage)
                if counts is None:
                    continue
                logger.info(
                    "run=%s stage=%s total=%s pending=%s running=%s succeeded=%s failed=%s skipped=%s",
                    snapshot.run_id,
                    stage.value,
                    counts.total_items,
                    counts.pending_items,
                    counts.running_items,
                    counts.completed_items,
                    counts.failed_items,
                    counts.skipped_items,
                )
            return
        if event.event_type in {"run_started", "stage_started", "run_finished"}:
            logger.info(message, *args)


class RichProgressRenderer:
    """Renders progress snapshots to the terminal using Rich."""

    def __init__(self) -> None:
        from rich.console import Console

        self.console = Console(stderr=True)
        self._live: Live | None = None

    def __call__(self, event: ProgressEvent) -> None:
        table = self._build_table(event.snapshot)
        if self.console.is_terminal:
            if self._live is None:
                from rich.live import Live

                self._live = Live(table, console=self.console, refresh_per_second=10)
                self._live.start()
            else:
                self._live.update(table, refresh=True)
            if event.event_type == "run_finished" and self._live is not None:
                self._live.stop()
                self._live = None
            return
        self.console.print(table)

    def _build_table(self, snapshot: RunProgressSnapshot) -> RenderableType:
        from rich.table import Table

        table = Table(
            title=(
                f"Run {snapshot.run_id} "
                f"(active={snapshot.active_stage.value if snapshot.active_stage else '-'})"
            )
        )
        table.add_column("Stage")
        table.add_column("Processed")
        table.add_column("Remaining")
        table.add_column("In Flight")
        table.add_column("Total")
        for stage in RunStage:
            counts = snapshot.stage_counts.get(stage)
            if counts is None:
                continue
            processed = (
                counts.completed_items + counts.failed_items + counts.skipped_items
            )
            remaining = counts.pending_items + counts.running_items
            table.add_row(
                stage.value,
                str(processed),
                str(remaining),
                str(counts.running_items),
                str(counts.total_items),
            )
        return table
