"""Interfaces (ports) that external adapters must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Protocol, Sequence

from themis.core import entities


class ModelProvider(ABC):
    """Abstract interface for anything capable of fulfilling generation tasks."""

    @abstractmethod
    def generate(
        self, task: entities.GenerationTask
    ) -> entities.GenerationRecord:  # pragma: no cover - abstract
        raise NotImplementedError


class DatasetAdapter(ABC):
    """Produces raw rows or fully built tasks for experiments."""

    @abstractmethod
    def iter_samples(self) -> Iterable[dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError


class Extractor(Protocol):
    def extract(self, raw_output: str) -> Any:  # pragma: no cover - protocol
        ...


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> entities.MetricScore:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = [
    "ModelProvider",
    "DatasetAdapter",
    "Extractor",
    "Metric",
]
