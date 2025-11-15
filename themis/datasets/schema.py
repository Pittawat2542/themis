"""Dataset schema and metadata definitions.

This module provides enhanced dataset abstractions with schema validation,
metadata, and filtering capabilities while maintaining backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@dataclass
class DatasetSchema:
    """Describes the structure and validation rules for dataset samples.

    Examples:
        # Basic schema
        schema = DatasetSchema(
            id_field="unique_id",
            reference_field="answer",
            required_fields={"unique_id", "problem", "answer"},
        )

        # Schema with validation
        def validate_problem(sample: dict) -> None:
            if len(sample.get("problem", "")) < 10:
                raise ValueError("Problem text too short")

        schema = DatasetSchema(
            id_field="id",
            reference_field="expected",
            required_fields={"id", "problem", "expected"},
            validators=[validate_problem],
        )
    """

    id_field: str
    reference_field: str | None
    required_fields: set[str] = field(default_factory=set)
    optional_fields: set[str] = field(default_factory=set)
    metadata_fields: set[str] = field(default_factory=set)
    validators: list[Callable[[dict], None]] = field(default_factory=list)

    def validate_sample(self, sample: dict[str, Any]) -> None:
        """Validate a single sample against this schema.

        Args:
            sample: Sample to validate

        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in sample:
                raise ValueError(
                    f"Missing required field '{field_name}' in sample {sample.get(self.id_field)}"
                )

        # Run custom validators
        for validator in self.validators:
            validator(sample)

    def get_all_fields(self) -> set[str]:
        """Get all known fields (required + optional + metadata)."""
        return self.required_fields | self.optional_fields | self.metadata_fields


@dataclass
class DatasetMetadata:
    """Metadata about the entire dataset.

    This provides information useful for experiment planning, reporting,
    and understanding dataset characteristics.

    Examples:
        metadata = DatasetMetadata(
            name="MATH-500",
            version="1.0",
            total_samples=500,
            categories={
                "subject": ["algebra", "geometry", "number_theory"],
                "difficulty": ["easy", "medium", "hard"],
            },
            difficulty_distribution={
                "easy": 100,
                "medium": 250,
                "hard": 150,
            },
            description="Math problems from competition mathematics",
        )
    """

    name: str
    version: str = "1.0"
    total_samples: int | None = None
    categories: dict[str, list[str]] = field(default_factory=dict)
    difficulty_distribution: dict[str, int] | None = None
    description: str = ""
    source_url: str | None = None
    license: str | None = None
    citation: str | None = None
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def get_category_values(self, category: str) -> list[str]:
        """Get all possible values for a category."""
        return self.categories.get(category, [])

    def has_category(self, category: str) -> bool:
        """Check if dataset has a specific category."""
        return category in self.categories


@runtime_checkable
class EnhancedDatasetAdapter(Protocol):
    """Extended dataset interface with schema and metadata support.

    This protocol extends the basic DatasetAdapter with additional
    capabilities for schema validation, filtering, and stratification.
    """

    def iter_samples(self) -> Iterable[dict[str, Any]]:
        """Iterate over dataset samples."""
        ...

    def get_schema(self) -> DatasetSchema:
        """Get the dataset schema."""
        ...

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        ...

    def filter(
        self, predicate: Callable[[dict[str, Any]], bool]
    ) -> EnhancedDatasetAdapter:
        """Return filtered view of dataset.

        Args:
            predicate: Function that returns True for samples to keep

        Returns:
            New dataset adapter with filtered samples
        """
        ...

    def limit(self, n: int) -> EnhancedDatasetAdapter:
        """Return dataset limited to first n samples.

        Args:
            n: Maximum number of samples

        Returns:
            New dataset adapter with limited samples
        """
        ...

    def stratify(
        self, field: str, distribution: dict[str, float]
    ) -> EnhancedDatasetAdapter:
        """Return stratified sample of dataset.

        Args:
            field: Field to stratify by
            distribution: Desired distribution (values should sum to 1.0)

        Returns:
            New dataset adapter with stratified samples
        """
        ...


# Common validators
def validate_non_empty_field(field_name: str) -> Callable[[dict], None]:
    """Create validator that ensures field is non-empty.

    Args:
        field_name: Name of field to validate

    Returns:
        Validator function
    """

    def validator(sample: dict) -> None:
        value = sample.get(field_name)
        if not value:
            raise ValueError(f"Field '{field_name}' cannot be empty")

    return validator


def validate_field_type(field_name: str, expected_type: type) -> Callable[[dict], None]:
    """Create validator that ensures field has correct type.

    Args:
        field_name: Name of field to validate
        expected_type: Expected type

    Returns:
        Validator function
    """

    def validator(sample: dict) -> None:
        value = sample.get(field_name)
        if value is not None and not isinstance(value, expected_type):
            raise ValueError(
                f"Field '{field_name}' expected type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    return validator


def validate_field_in_choices(
    field_name: str, choices: set[str]
) -> Callable[[dict], None]:
    """Create validator that ensures field value is in allowed choices.

    Args:
        field_name: Name of field to validate
        choices: Set of allowed values

    Returns:
        Validator function
    """

    def validator(sample: dict) -> None:
        value = sample.get(field_name)
        if value is not None and value not in choices:
            raise ValueError(
                f"Field '{field_name}' value '{value}' not in allowed choices: {choices}"
            )

    return validator


__all__ = [
    "DatasetSchema",
    "DatasetMetadata",
    "EnhancedDatasetAdapter",
    "validate_non_empty_field",
    "validate_field_type",
    "validate_field_in_choices",
]
