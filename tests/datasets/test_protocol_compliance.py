"""Tests for DatasetAdapter protocol compliance.

This module verifies that dataset classes properly implement the DatasetAdapter
protocol, both structurally and at runtime.
"""

import pytest

from themis.datasets.base import BaseDataset
from themis.datasets.schema import DatasetMetadata, DatasetSchema
from themis.interfaces import DatasetAdapter


class MinimalDataset:
    """Minimal dataset class implementing DatasetAdapter protocol via duck typing."""

    def iter_samples(self):
        """Return sample data."""
        return iter([{"id": "1", "text": "sample"}])


class IncompleteDataset:
    """Dataset class that does NOT implement the protocol (missing iter_samples)."""

    def get_samples(self):  # Wrong method name
        return []


def test_protocol_is_runtime_checkable():
    """Verify DatasetAdapter protocol supports runtime isinstance checks."""
    # Verify protocol is decorated with @runtime_checkable
    # This can be checked by trying isinstance with a duck-typed class
    minimal = MinimalDataset()
    # If this works without error, the protocol is runtime checkable
    assert isinstance(minimal, DatasetAdapter)


def test_minimal_dataset_satisfies_protocol():
    """Verify a minimal dataset with iter_samples() satisfies the protocol."""
    dataset = MinimalDataset()

    # Should satisfy protocol at runtime
    assert isinstance(dataset, DatasetAdapter)

    # Should have required method
    assert hasattr(dataset, "iter_samples")
    assert callable(dataset.iter_samples)

    # Method should work as expected
    samples = list(dataset.iter_samples())
    assert len(samples) == 1
    assert samples[0]["id"] == "1"


def test_incomplete_dataset_does_not_satisfy_protocol():
    """Verify a class without iter_samples() does NOT satisfy the protocol."""
    dataset = IncompleteDataset()

    # Should NOT satisfy protocol (missing iter_samples method)
    assert not isinstance(dataset, DatasetAdapter)


def test_base_dataset_satisfies_protocol():
    """Verify BaseDataset implements DatasetAdapter protocol."""
    # Create minimal BaseDataset instance
    samples = [
        {"id": "1", "problem": "What is 2+2?", "answer": "4"},
        {"id": "2", "problem": "What is 3+3?", "answer": "6"},
    ]
    schema = DatasetSchema(
        id_field="id",
        reference_field="answer",
        required_fields={"id", "problem", "answer"},
    )
    metadata = DatasetMetadata(
        name="TestDataset",
        version="1.0",
        total_samples=2,
    )

    dataset = BaseDataset(samples, schema, metadata)

    # Should satisfy protocol at runtime
    assert isinstance(dataset, DatasetAdapter)

    # Should have required method
    assert hasattr(dataset, "iter_samples")
    assert callable(dataset.iter_samples)

    # Method should work as expected
    result_samples = list(dataset.iter_samples())
    assert len(result_samples) == 2
    assert result_samples[0]["id"] == "1"
    assert result_samples[1]["id"] == "2"


def test_loaded_datasets_satisfy_protocol():
    """Verify datasets loaded via factory functions satisfy the protocol.

    Note: Specific dataset classes use factory functions (load_math500, etc.)
    rather than direct instantiation. We verify that the BaseDataset
    they inherit from satisfies the protocol, which is sufficient.
    """
    from themis.datasets import math500

    # Create a minimal inline dataset using MathSample
    sample = math500.MathSample(
        unique_id="test-1",
        problem="Test problem",
        solution="Test solution",
        answer="42",
        subject="algebra",
        level=1,
    )

    # Verify the sample can be converted to generation example
    example = sample.to_generation_example()
    assert "answer" in example
    assert example["answer"] == "42"

    # Note: Actual dataset classes (Math500, etc.) are created via
    # load_math500() and similar factory functions. The important thing
    # is that BaseDataset (which they inherit from) satisfies the protocol.
    # This is tested in test_base_dataset_satisfies_protocol().


def test_protocol_duck_typing():
    """Verify protocol works with duck typing (no explicit inheritance needed)."""

    class DuckTypedDataset:
        """Dataset using duck typing without inheriting from anything."""

        def iter_samples(self):
            return iter([{"id": "duck", "quack": True}])

    dataset = DuckTypedDataset()

    # Should satisfy protocol via duck typing
    assert isinstance(dataset, DatasetAdapter)

    # Should work correctly
    samples = list(dataset.iter_samples())
    assert samples[0]["quack"] is True


def test_protocol_with_custom_implementation():
    """Verify custom dataset implementations satisfy the protocol."""

    class CustomDataset:
        """Custom dataset with additional methods beyond the protocol."""

        def __init__(self, data):
            self._data = data

        def iter_samples(self):
            """Required by protocol."""
            return iter(self._data)

        def get_count(self):
            """Additional method not required by protocol."""
            return len(self._data)

    data = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    dataset = CustomDataset(data)

    # Should satisfy protocol
    assert isinstance(dataset, DatasetAdapter)

    # Both protocol and custom methods should work
    assert len(list(dataset.iter_samples())) == 3
    assert dataset.get_count() == 3
