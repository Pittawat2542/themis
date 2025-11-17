"""Edge case tests for DatasetRegistry.

This module tests boundary conditions, error handling, and edge cases
for the DatasetRegistry to ensure robust operation.
"""

import pytest

from themis.datasets.registry import (
    DatasetFactory,
    DatasetRegistry,
    create_dataset,
    is_dataset_registered,
    list_datasets,
    register_dataset,
)


def test_registry_duplicate_registration_raises():
    """Test that re-registering a dataset raises ValueError."""
    registry = DatasetRegistry()

    def factory1(options):
        return [{"id": "1", "version": "v1"}]

    def factory2(options):
        return [{"id": "1", "version": "v2"}]

    # Register first factory
    registry.register("test-dataset", factory1)

    # Try to re-register - should raise
    with pytest.raises(ValueError, match="already registered"):
        registry.register("test-dataset", factory2)


def test_registry_create_non_existent_dataset_raises():
    """Test creating a non-existent dataset raises ValueError."""
    registry = DatasetRegistry()

    with pytest.raises(ValueError, match="Unknown dataset"):
        registry.create("nonexistent")


def test_registry_factory_that_returns_empty_list():
    """Test factory that returns empty list works correctly."""
    registry = DatasetRegistry()

    def empty_factory(options):
        return []

    registry.register("empty-dataset", empty_factory)
    result = registry.create("empty-dataset")

    assert result == []
    assert isinstance(result, list)


def test_registry_factory_that_raises_exception():
    """Test factory that raises exception propagates error."""
    registry = DatasetRegistry()

    def failing_factory(options):
        raise RuntimeError("Factory failed!")

    registry.register("failing-dataset", failing_factory)

    with pytest.raises(RuntimeError, match="Factory failed!"):
        registry.create("failing-dataset")


def test_registry_factory_with_string_return():
    """Test factory that returns a string (iterable but not list)."""
    registry = DatasetRegistry()

    def string_factory(options):
        return "abc"  # String is iterable

    registry.register("string-dataset", string_factory)

    # Will return the string as-is (no validation in registry)
    result = registry.create("string-dataset")
    # This demonstrates the registry doesn't validate return types
    assert result == "abc"


def test_registry_factory_receives_all_options():
    """Test factory receives all provided options."""
    registry = DatasetRegistry()
    received_options = {}

    def capture_options(options):
        received_options.update(options)
        return []

    registry.register("capture-dataset", capture_options)
    registry.create(
        "capture-dataset",
        limit=10,
        source="test",
        custom_param="value",
    )

    assert received_options["limit"] == 10
    assert received_options["source"] == "test"
    assert received_options["custom_param"] == "value"


def test_registry_factory_with_no_options():
    """Test factory called with empty options dict."""
    registry = DatasetRegistry()

    def no_options_factory(options):
        assert options == {}
        return [{"id": "1"}]

    registry.register("no-options", no_options_factory)
    result = registry.create("no-options")

    assert len(result) == 1


def test_registry_list_datasets_empty():
    """Test listing datasets when registry is empty."""
    registry = DatasetRegistry()

    datasets = registry.list_datasets()
    assert datasets == []


def test_registry_list_datasets_returns_sorted():
    """Test list_datasets returns sorted list."""
    registry = DatasetRegistry()

    # Register in random order
    registry.register("zebra", lambda opts: [])
    registry.register("alpha", lambda opts: [])
    registry.register("beta", lambda opts: [])

    datasets = registry.list_datasets()

    # Should be sorted alphabetically
    assert datasets == ["alpha", "beta", "zebra"]


def test_registry_is_registered_checks():
    """Test is_registered method works correctly."""
    registry = DatasetRegistry()

    assert not registry.is_registered("test")

    registry.register("test", lambda opts: [])

    assert registry.is_registered("test")
    assert not registry.is_registered("other")


def test_global_registry_functions():
    """Test global registry functions work correctly."""
    # Note: These affect global state, so tests may interfere

    # Register a unique dataset
    unique_name = "test-global-registry-12345"

    def factory(opts):
        return [{"id": "test"}]

    # Should not be registered initially
    assert not is_dataset_registered(unique_name)

    # Register it
    register_dataset(unique_name, factory)

    # Should now be registered
    assert is_dataset_registered(unique_name)

    # Should appear in list
    datasets = list_datasets()
    assert unique_name in datasets

    # Should be createable
    result = create_dataset(unique_name)
    assert len(result) == 1
    assert result[0]["id"] == "test"


def test_registry_with_callable_class():
    """Test registry works with callable class as factory."""
    registry = DatasetRegistry()

    class DatasetFactory:
        def __call__(self, options):
            limit = options.get("limit", 3)
            return [{"id": str(i)} for i in range(limit)]

    factory_instance = DatasetFactory()
    registry.register("class-factory", factory_instance)

    result = registry.create("class-factory", limit=5)

    assert len(result) == 5
    assert result[0]["id"] == "0"
    assert result[4]["id"] == "4"


def test_registry_with_lambda_factory():
    """Test registry works with lambda as factory."""
    registry = DatasetRegistry()

    registry.register("lambda-dataset", lambda opts: [{"id": "lambda"}])

    result = registry.create("lambda-dataset")

    assert len(result) == 1
    assert result[0]["id"] == "lambda"


def test_registry_dataset_name_case_sensitivity():
    """Test dataset names are case-sensitive."""
    registry = DatasetRegistry()

    registry.register("MyDataset", lambda opts: [{"version": "upper"}])
    registry.register("mydataset", lambda opts: [{"version": "lower"}])

    result_upper = registry.create("MyDataset")
    result_lower = registry.create("mydataset")

    # Should be different datasets
    assert result_upper[0]["version"] == "upper"
    assert result_lower[0]["version"] == "lower"


def test_registry_with_special_characters_in_name():
    """Test registry handles dataset names with special characters."""
    registry = DatasetRegistry()

    # Register datasets with various special characters
    special_names = [
        "dataset-with-hyphens",
        "dataset_with_underscores",
        "dataset.with.dots",
        "dataset123",
        "UPPERCASE",
    ]

    for name in special_names:
        registry.register(name, lambda opts: [{"name": name}])

    # All should be registered
    for name in special_names:
        assert registry.is_registered(name)


def test_registry_factory_with_default_parameters():
    """Test factory that uses default parameters."""
    registry = DatasetRegistry()

    def factory_with_defaults(options):
        limit = options.get("limit", 10)  # Default to 10
        prefix = options.get("prefix", "item")  # Default prefix
        return [{"id": f"{prefix}-{i}"} for i in range(limit)]

    registry.register("defaults-dataset", factory_with_defaults)

    # Test with defaults
    result1 = registry.create("defaults-dataset")
    assert len(result1) == 10
    assert result1[0]["id"] == "item-0"

    # Test with overrides
    result2 = registry.create("defaults-dataset", limit=3, prefix="test")
    assert len(result2) == 3
    assert result2[0]["id"] == "test-0"


def test_registry_multiple_registrations_in_sequence():
    """Test registering many datasets in sequence works."""
    registry = DatasetRegistry()

    # Register 100 datasets
    for i in range(100):
        registry.register(f"dataset-{i}", lambda opts, idx=i: [{"id": idx}])

    # All should be registered
    assert len(registry.list_datasets()) == 100

    # Spot check some
    assert registry.is_registered("dataset-0")
    assert registry.is_registered("dataset-50")
    assert registry.is_registered("dataset-99")


def test_registry_create_with_none_options():
    """Test create handles None in options gracefully."""
    registry = DatasetRegistry()

    def factory(options):
        # Factory should handle None values in options
        value = options.get("nullable_param")
        return [{"value": value}]

    registry.register("nullable-dataset", factory)

    result = registry.create("nullable-dataset", nullable_param=None)

    assert result[0]["value"] is None
