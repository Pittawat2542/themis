"""Registry for dataset loaders.

This module provides a plugin-based registry system for datasets, allowing
users to register custom datasets without modifying core Themis code.

Example:
    ```python
    from themis.datasets import register_dataset

    def create_my_dataset(**options):
        from my_module import MyDataset
        return MyDataset(path=options.get('path'))

    register_dataset('my-dataset', create_my_dataset)
    ```
"""

from __future__ import annotations

from typing import Any, Callable

# Factory type: takes config options, returns list of samples
DatasetFactory = Callable[[dict[str, Any]], list[dict[str, Any]]]


class DatasetRegistry:
    """Registry for dataset loaders.

    Maintains a mapping from dataset names to factory functions that
    load and return dataset samples.
    """

    def __init__(self):
        self._datasets: dict[str, DatasetFactory] = {}

    def register(self, name: str, factory: DatasetFactory) -> None:
        """Register a dataset factory.

        Args:
            name: Unique identifier for the dataset (e.g., 'math500', 'my-dataset')
            factory: Callable that takes config options and returns list of samples

        Raises:
            ValueError: If dataset name is already registered
        """
        if name in self._datasets:
            raise ValueError(
                f"Dataset '{name}' is already registered. "
                f"Use a different name or unregister the existing dataset first."
            )
        self._datasets[name] = factory

    def unregister(self, name: str) -> None:
        """Unregister a dataset.

        Args:
            name: Dataset identifier to remove

        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in self._datasets:
            raise ValueError(f"Dataset '{name}' is not registered")
        del self._datasets[name]

    def create(self, name: str, **options) -> list[dict[str, Any]]:
        """Create a dataset instance by loading samples.

        Args:
            name: Registered dataset identifier
            **options: Configuration options passed to the factory function
                Common options include:
                - source: 'huggingface', 'local', or custom source
                - data_dir: Path for local datasets
                - split: Dataset split (e.g., 'train', 'test')
                - limit: Maximum number of samples to load
                - subjects: List of subjects to filter

        Returns:
            List of sample dictionaries ready for generation

        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in self._datasets:
            available = list(self._datasets.keys())
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available datasets: {', '.join(sorted(available)) or 'none'}"
            )
        factory = self._datasets[name]
        return factory(options)

    def list_datasets(self) -> list[str]:
        """List all registered dataset names.

        Returns:
            Sorted list of registered dataset identifiers
        """
        return sorted(self._datasets.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: Dataset identifier to check

        Returns:
            True if the dataset is registered, False otherwise
        """
        return name in self._datasets


# Global registry instance
_REGISTRY = DatasetRegistry()


def register_dataset(name: str, factory: DatasetFactory) -> None:
    """Register a dataset factory in the global registry.

    Args:
        name: Unique identifier for the dataset
        factory: Callable that takes config options and returns samples

    Example:
        ```python
        def create_my_dataset(options):
            from my_module import load_data
            return load_data(
                path=options.get('path'),
                limit=options.get('limit')
            )

        register_dataset('my-dataset', create_my_dataset)
        ```
    """
    _REGISTRY.register(name, factory)


def unregister_dataset(name: str) -> None:
    """Unregister a dataset from the global registry.

    Args:
        name: Dataset identifier to remove
    """
    _REGISTRY.unregister(name)


def create_dataset(name: str, **options) -> list[dict[str, Any]]:
    """Create a dataset by loading samples from a registered factory.

    Args:
        name: Registered dataset identifier
        **options: Configuration options for the dataset

    Returns:
        List of sample dictionaries

    Example:
        ```python
        samples = create_dataset(
            'math500',
            source='huggingface',
            split='test',
            limit=10
        )
        ```
    """
    return _REGISTRY.create(name, **options)


def list_datasets() -> list[str]:
    """List all registered datasets.

    Returns:
        Sorted list of dataset names
    """
    return _REGISTRY.list_datasets()


def is_dataset_registered(name: str) -> bool:
    """Check if a dataset is registered.

    Args:
        name: Dataset identifier

    Returns:
        True if registered, False otherwise
    """
    return _REGISTRY.is_registered(name)


__all__ = [
    "DatasetFactory",
    "DatasetRegistry",
    "register_dataset",
    "unregister_dataset",
    "create_dataset",
    "list_datasets",
    "is_dataset_registered",
]
