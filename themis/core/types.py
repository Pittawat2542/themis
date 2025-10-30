"""Common type definitions and generic types for Themis.

This module provides improved type safety through generic types and protocols.
All types are designed to be backward compatible with existing code.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from themis.core import entities

# Type variables for generic types
T = TypeVar("T")  # Generic type for predictions/references
T_co = TypeVar("T_co", covariant=True)  # Covariant type for outputs


@runtime_checkable
class TypedExtractor(Protocol[T_co]):
    """Protocol for extractors with typed output.

    This is a backward-compatible extension of the Extractor protocol that
    provides type information about the extraction output.
    """

    def extract(self, raw_output: str) -> T_co:
        """Extract structured data from raw output.

        Args:
            raw_output: Raw text output from model

        Returns:
            Extracted value of type T_co

        Raises:
            FieldExtractionError: If extraction fails
        """
        ...


@runtime_checkable
class TypedMetric(Protocol[T]):
    """Protocol for metrics with typed predictions.

    This is a backward-compatible extension of the Metric interface that
    provides type information about expected prediction types.
    """

    name: str

    def compute(
        self,
        *,
        prediction: T,
        references: Sequence[T],
        metadata: dict[str, Any] | None = None,
    ) -> entities.MetricScore:
        """Compute metric score.

        Args:
            prediction: Model prediction of type T
            references: Reference answers of type T
            metadata: Optional metadata

        Returns:
            MetricScore with computed value
        """
        ...


# Common type aliases for better readability
PredictionType = TypeVar("PredictionType")
ReferenceType = TypeVar("ReferenceType")
ExtractionType = TypeVar("ExtractionType")


class ValidationError(ValueError):
    """Raised when runtime type validation fails."""

    pass


def validate_type(value: Any, expected_type: type[T], field_name: str = "value") -> T:
    """Validate value against expected type at runtime.

    Args:
        value: Value to validate
        expected_type: Expected type
        field_name: Name of field for error messages

    Returns:
        Value cast to expected type

    Raises:
        ValidationError: If type validation fails
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{field_name} expected type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def validate_sequence_type(
    values: Sequence[Any], expected_type: type[T], field_name: str = "values"
) -> Sequence[T]:
    """Validate all values in sequence against expected type.

    Args:
        values: Sequence to validate
        expected_type: Expected type for elements
        field_name: Name of field for error messages

    Returns:
        Validated sequence

    Raises:
        ValidationError: If any element fails validation
    """
    for i, value in enumerate(values):
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"{field_name}[{i}] expected type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
    return values


__all__ = [
    "T",
    "T_co",
    "TypedExtractor",
    "TypedMetric",
    "PredictionType",
    "ReferenceType",
    "ExtractionType",
    "ValidationError",
    "validate_type",
    "validate_sequence_type",
]
