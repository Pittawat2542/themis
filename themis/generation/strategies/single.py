"""Default strategy - run exactly once and pass-through result."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from themis.core import entities as core_entities


@dataclass
class SingleAttemptStrategy:
    """Default strategy – run exactly once and pass-through result."""

    def expand(
        self, task: core_entities.GenerationTask
    ) -> Iterable[core_entities.GenerationTask]:
        return [task]

    def aggregate(
        self,
        task: core_entities.GenerationTask,
        records: list[core_entities.GenerationRecord],
    ) -> core_entities.GenerationRecord:
        record = records[0]
        return core_entities.GenerationRecord(
            task=task,
            output=record.output,
            error=record.error,
            metrics=dict(record.metrics),
        )
