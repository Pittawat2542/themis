"""Tests for dataset registry system."""

from __future__ import annotations

import pytest

from themis.datasets import registry


def test_registry_creation():
    """Test creating a new registry instance."""
    reg = registry.DatasetRegistry()
    assert reg.list_datasets() == []
    assert not reg.is_registered("test")


def test_register_dataset():
    """Test registering a dataset."""
    reg = registry.DatasetRegistry()

    def factory(options):
        return [{"id": "1", "text": "test"}]

    reg.register("test-dataset", factory)
    assert reg.is_registered("test-dataset")
    assert "test-dataset" in reg.list_datasets()


def test_register_duplicate_dataset_raises():
    """Test that registering duplicate dataset name raises error."""
    reg = registry.DatasetRegistry()

    def factory(options):
        return []

    reg.register("test", factory)

    with pytest.raises(ValueError, match="already registered"):
        reg.register("test", factory)


def test_unregister_dataset():
    """Test unregistering a dataset."""
    reg = registry.DatasetRegistry()

    def factory(options):
        return []

    reg.register("test", factory)
    assert reg.is_registered("test")

    reg.unregister("test")
    assert not reg.is_registered("test")
    assert "test" not in reg.list_datasets()


def test_unregister_nonexistent_dataset_raises():
    """Test that unregistering nonexistent dataset raises error."""
    reg = registry.DatasetRegistry()

    with pytest.raises(ValueError, match="not registered"):
        reg.unregister("nonexistent")


def test_create_dataset():
    """Test creating a dataset from registry."""
    reg = registry.DatasetRegistry()

    def factory(options):
        limit = options.get("limit", 10)
        return [{"id": str(i)} for i in range(limit)]

    reg.register("test", factory)

    # Create with default limit
    samples = reg.create("test")
    assert len(samples) == 10

    # Create with custom limit
    samples = reg.create("test", limit=5)
    assert len(samples) == 5


def test_create_nonexistent_dataset_raises():
    """Test that creating nonexistent dataset raises error."""
    reg = registry.DatasetRegistry()

    with pytest.raises(ValueError, match="Unknown dataset"):
        reg.create("nonexistent")


def test_list_datasets_sorted():
    """Test that list_datasets returns sorted list."""
    reg = registry.DatasetRegistry()

    def factory(options):
        return []

    reg.register("zebra", factory)
    reg.register("apple", factory)
    reg.register("middle", factory)

    datasets = reg.list_datasets()
    assert datasets == ["apple", "middle", "zebra"]


def test_global_registry_has_builtin_datasets():
    """Test that global registry has all built-in datasets."""
    from themis.datasets import list_datasets

    datasets = list_datasets()

    # Check for key built-in datasets
    assert "math500" in datasets
    assert "supergpqa" in datasets
    assert "mmlu-pro" in datasets

    # Check for competition datasets
    assert "aime24" in datasets
    assert "aime25" in datasets
    assert "amc23" in datasets
    assert "olympiadbench" in datasets
    assert "beyondaime" in datasets
    assert "gsm8k" in datasets
    assert "gpqa" in datasets


def test_global_registry_functions():
    """Test global registry convenience functions."""
    from themis.datasets import (
        is_dataset_registered,
        list_datasets,
        register_dataset,
        unregister_dataset,
    )

    # Test listing
    datasets = list_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0

    # Test checking
    assert is_dataset_registered("math500")
    assert not is_dataset_registered("totally-fake-dataset-xyz")

    # Test registering custom dataset
    def custom_factory(options):
        return [{"custom": True}]

    register_dataset("test-custom", custom_factory)
    assert is_dataset_registered("test-custom")

    # Test unregistering
    unregister_dataset("test-custom")
    assert not is_dataset_registered("test-custom")


def test_dataset_factory_receives_options():
    """Test that dataset factory receives all options."""
    reg = registry.DatasetRegistry()

    received_options = {}

    def factory(options):
        received_options.update(options)
        return []

    reg.register("test", factory)
    reg.create(
        "test",
        source="huggingface",
        limit=10,
        subjects=["math"],
        custom_param="value",
    )

    assert received_options["source"] == "huggingface"
    assert received_options["limit"] == 10
    assert received_options["subjects"] == ["math"]
    assert received_options["custom_param"] == "value"


def test_registry_isolation():
    """Test that different registry instances are isolated."""
    reg1 = registry.DatasetRegistry()
    reg2 = registry.DatasetRegistry()

    def factory(options):
        return []

    reg1.register("test", factory)

    assert reg1.is_registered("test")
    assert not reg2.is_registered("test")
