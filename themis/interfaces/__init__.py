"""Interfaces (ports) that external adapters must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from themis.core import entities


class ModelProvider(ABC):
    """Abstract interface for anything capable of fulfilling generation tasks."""

    @abstractmethod
    def generate(
        self, task: entities.GenerationTask
    ) -> entities.GenerationRecord:  # pragma: no cover - abstract
        raise NotImplementedError


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol for dataset adapters that produce raw samples for experiments.

    This is a structural protocol that can be satisfied by any class implementing
    the required methods, without explicit inheritance. The @runtime_checkable
    decorator allows isinstance() checks at runtime.

    Required Methods:
        iter_samples: Returns an iterable of sample dictionaries

    Example:
        >>> class MyDataset:
        ...     def iter_samples(self):
        ...         return iter([{"id": "1", "text": "sample"}])
        ...
        >>> isinstance(MyDataset(), DatasetAdapter)  # True at runtime

    Note:
        Classes do not need to explicitly inherit from this protocol.
        Duck typing is sufficient - any class with an iter_samples() method
        will be recognized as a DatasetAdapter at runtime.
    """

    def iter_samples(self) -> Iterable[dict[str, Any]]:  # pragma: no cover - protocol
        """Iterate over dataset samples.

        Returns:
            Iterable of dictionaries, each representing a dataset sample

        Example:
            >>> for sample in dataset.iter_samples():
            ...     print(sample["id"])
        """
        ...


class Extractor(Protocol):
    def extract(self, raw_output: str) -> Any:  # pragma: no cover - protocol
        ...


class Metric(ABC):
    name: str
    requires_reference: bool = True

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
