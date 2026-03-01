"""Repeat the same task multiple times for test-time scaling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from themis.core import entities as core_entities


@dataclass
class RepeatedSamplingStrategy:
    """Repeat the same task multiple times for test-time scaling."""

    attempts: int
    metadata_label: str = "attempts"

    def expand(
        self, task: core_entities.GenerationTask
    ) -> Iterable[core_entities.GenerationTask]:
        for index in range(self.attempts):
            attempt_metadata = dict(task.metadata)
            attempt_metadata[self.metadata_label] = index
            yield core_entities.GenerationTask(
                prompt=task.prompt,
                model=task.model,
                sampling=task.sampling,
                metadata=attempt_metadata,
                reference=task.reference,
            )

    def aggregate(
        self,
        task: core_entities.GenerationTask,
        records: list[core_entities.GenerationRecord],
    ) -> core_entities.GenerationRecord:
        best = next((record for record in records if not record.error), records[0])
        aggregated = core_entities.GenerationRecord(
            task=task,
            output=best.output,
            error=best.error,
            metrics=dict(best.metrics),
        )
        aggregated.metrics["attempt_count"] = len(records)
        aggregated.metrics["attempt_outcomes"] = [
            {
                "output": record.output.text if record.output else None,
                "error": record.error.message if record.error else None,
            }
            for record in records
        ]
        return aggregated
