"""Base interface for generation strategies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from themis.core import entities as core_entities


class GenerationStrategy(Protocol):
    """Strategy responsible for expanding a task into one or more execution attempts."""

    def expand(
        self, task: core_entities.GenerationTask
    ) -> Iterable[core_entities.GenerationTask]:  # pragma: no cover - interface
        ...

    def aggregate(
        self,
        task: core_entities.GenerationTask,
        records: list[core_entities.GenerationRecord],
    ) -> core_entities.GenerationRecord:  # pragma: no cover - interface
        ...
