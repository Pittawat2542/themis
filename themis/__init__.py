"""Themis experiment platform - Dead simple LLM evaluation.

The primary interface is the `evaluate()` function:

    import themis
    report = themis.evaluate("math500", model="gpt-4", limit=100)

Extension APIs for registering custom components:
    - themis.register_metric() - Register custom metrics
    - themis.register_dataset() - Register custom datasets
    - themis.register_provider() - Register custom model providers
    - themis.register_benchmark() - Register custom benchmark presets
"""

from themis._version import __version__
from themis.api import evaluate, get_registered_metrics, register_metric
from themis.datasets import register_dataset, list_datasets, is_dataset_registered
from themis.presets import register_benchmark, list_benchmarks, get_benchmark_preset
from themis.providers import register_provider, list_providers
from themis.exceptions import (
    ThemisError,
    ConfigurationError,
    ProviderError,
    DatasetError,
    MetricError,
    EvaluationError,
    StorageError,
)


def __getattr__(name: str):
    """Lazy-load heavy submodules on first access."""
    import importlib

    _LAZY_SUBMODULES = {"config", "core", "evaluation", "generation"}
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"themis.{name}")
    raise AttributeError(f"module 'themis' has no attribute {name!r}")


__all__ = [
    # Main API
    "evaluate",
    # Metrics
    "register_metric",
    "get_registered_metrics",
    # Datasets
    "register_dataset",
    "list_datasets",
    "is_dataset_registered",
    # Benchmarks
    "register_benchmark",
    "list_benchmarks",
    "get_benchmark_preset",
    # Providers
    "register_provider",
    "list_providers",
    # Exceptions
    "ThemisError",
    "ConfigurationError",
    "ProviderError",
    "DatasetError",
    "MetricError",
    "EvaluationError",
    "StorageError",
    # Submodules (lazy)
    "config",
    "core",
    "evaluation",
    "generation",
    # Version
    "__version__",
]
