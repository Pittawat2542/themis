"""Custom generation runner for the advanced experiment."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from themis.core import entities as core_entities
from themis.generation import router as base_router
from themis.generation import runner as base_runner


class PrioritizedGenerationRunner(base_runner.GenerationRunner):
    """Batches generation tasks by subject and model to reduce switching costs."""

    def __init__(
        self,
        *,
        provider,
        priority_field: str = "subject",
        chunk_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(provider=provider, **kwargs)
        self._priority_field = priority_field
        self._chunk_size = max(1, chunk_size)

    def run(
        self, tasks: Iterable[core_entities.GenerationTask]
    ) -> Iterator[core_entities.GenerationRecord]:  # type: ignore[override]
        task_list = list(tasks)
        task_list.sort(key=self._priority_key)
        for chunk in _chunk(task_list, self._chunk_size):
            for record in super().run(chunk):
                yield record

    def _priority_key(self, task: core_entities.GenerationTask):
        subject = str(task.metadata.get(self._priority_field, ""))
        model = task.model.identifier
        dataset_id = str(task.metadata.get("dataset_id", ""))
        return (subject, model, dataset_id)


def _chunk(sequence: Sequence[core_entities.GenerationTask], size: int):
    for index in range(0, len(sequence), size):
        yield sequence[index : index + size]


__all__ = ["PrioritizedGenerationRunner"]
# Optional router mixin for instrumentation


class TrackingProviderRouter(base_router.ProviderRouter):
    def __init__(self, providers):
        super().__init__(providers)
        self.call_history: list[str] = []

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        self.call_history.append(
            f"{task.metadata.get('subject', 'unknown')}::{task.model.identifier}"
        )
        return super().generate(task)


__all__.append("TrackingProviderRouter")
