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
    """Protocol for extractors that parse model output.

    Extractors are responsible for parsing raw model output text and
    extracting the relevant answer or prediction. The evaluation pipeline
    calls the extractor before passing the result to metrics.

    Example:
        >>> class JsonExtractor:
        ...     def extract(self, raw_output: str) -> Any:
        ...         import json
        ...         return json.loads(raw_output)["answer"]
    """

    def extract(self, raw_output: str) -> Any:  # pragma: no cover - protocol
        """Extract prediction from raw model output.

        Args:
            raw_output: Raw text output from the model

        Returns:
            Extracted prediction (type depends on extractor implementation)

        Raises:
            FieldExtractionError: If extraction fails
        """
        ...


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Metrics compute scores by comparing model predictions against reference values.
    The evaluation pipeline handles extraction before passing data to metrics.

    IMPORTANT - Extractor Contract:
        The 'prediction' parameter receives EXTRACTED output from the extractor,
        NOT raw model output. Metrics should NOT attempt to re-extract or parse
        the prediction - it has already been processed by the pipeline's extractor.

        Example flow:
            1. Model generates: "<think>reasoning</think><answer>42</answer>"
            2. Extractor extracts: "42"
            3. Metric receives: prediction="42" (already extracted)

    Attributes:
        name: Unique metric identifier
        requires_reference: Whether metric needs reference values (default: True)

    Example:
        >>> class ExactMatch(Metric):
        ...     name = "exact_match"
        ...
        ...     def compute(self, *, prediction, references, metadata=None):
        ...         # prediction is already extracted - no parsing needed
        ...         is_correct = any(prediction == ref for ref in references)
        ...         return MetricScore(
        ...             metric_name=self.name,
        ...             value=1.0 if is_correct else 0.0
        ...         )
    """

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
        """Compute metric score.

        Args:
            prediction: Extracted prediction from model output (already processed
                by extractor - do NOT re-extract or parse). Type depends on the
                extractor used in the pipeline.
            references: List of reference values in normalized format. Each element
                can be:
                - A scalar value (str, int, float, bool)
                - A dict (for multi-value references like {"target": 122, "numbers": [...]})
                - Any other type from the original reference
            metadata: Optional metadata dict containing:
                - "sample_id": Sample identifier (if available)
                - Additional task-specific metadata

        Returns:
            MetricScore with computed value and optional details

        Note:
            The prediction parameter is already extracted by the pipeline's extractor.
            Metrics should work with the extracted value directly, not attempt to
            parse or extract again from raw output.

        Example:
            >>> def compute(self, *, prediction, references, metadata=None):
            ...     # prediction is already extracted (e.g., "42", not "<answer>42</answer>")
            ...     # references is a list (e.g., ["42"] or [{"target": 42, "numbers": [...]}])
            ...     score_value = self._compare(prediction, references)
            ...     return MetricScore(metric_name=self.name, value=score_value)
        """
        raise NotImplementedError


__all__ = [
    "ModelProvider",
    "DatasetAdapter",
    "Extractor",
    "Metric",
]
