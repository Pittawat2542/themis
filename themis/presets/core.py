"""Core benchmark preset functionality and registry.

This module provides the base BenchmarkPreset dataclass and the
registry for managing available benchmark configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from themis.exceptions import ConfigurationError
from themis.interfaces import Extractor, Metric
from themis.generation.templates import PromptTemplate


@dataclass
class BenchmarkPreset:
    """Configuration preset for a benchmark.

    Attributes:
        name: Benchmark name
        prompt_template: Default prompt template
        metrics: List of metric instances
        extractor: Output extractor
        dataset_loader: Function to load the dataset
        metadata_fields: Fields to include in task metadata
        reference_field: Field containing the reference answer
        dataset_id_field: Field containing the sample ID
        description: Human-readable description
    """

    name: str
    prompt_template: PromptTemplate
    metrics: list[Metric]
    extractor: Extractor
    dataset_loader: Callable[[int | None], Sequence[dict[str, Any]]]
    metadata_fields: tuple[str, ...] = field(default_factory=tuple)
    reference_field: str = "answer"
    dataset_id_field: str = "id"
    description: str = ""

    def load_dataset(self, limit: int | None = None) -> Sequence[dict[str, Any]]:
        """Load the benchmark dataset.

        Args:
            limit: Maximum number of samples to load

        Returns:
            List of dataset samples
        """
        return self.dataset_loader(limit)


# Registry of benchmark presets
_BENCHMARK_REGISTRY: dict[str, BenchmarkPreset] = {}
_REGISTRY_INITIALIZED = False


def _ensure_registry_initialized() -> None:
    """Initialize benchmark registry on first use (lazy loading)."""
    global _REGISTRY_INITIALIZED
    if not _REGISTRY_INITIALIZED:
        from themis.presets.math_benchmarks import _register_math_benchmarks
        from themis.presets.mcq_benchmarks import _register_mcq_benchmarks
        from themis.presets.benchmarks import _register_demo_benchmark

        _register_math_benchmarks()
        _register_mcq_benchmarks()
        _register_demo_benchmark()
        _REGISTRY_INITIALIZED = True


def register_benchmark(preset: BenchmarkPreset) -> None:
    """Register a benchmark preset.

    Args:
        preset: Benchmark preset configuration
    """
    _BENCHMARK_REGISTRY[preset.name.lower()] = preset


def get_benchmark_preset(name: str) -> BenchmarkPreset:
    """Get a benchmark preset by name.

    Args:
        name: Benchmark name (case-insensitive)

    Returns:
        Benchmark preset

    Raises:
        ConfigurationError: If benchmark is not found
    """
    _ensure_registry_initialized()

    name_lower = name.lower()
    if name_lower not in _BENCHMARK_REGISTRY:
        available = ", ".join(sorted(_BENCHMARK_REGISTRY.keys()))
        raise ConfigurationError(
            f"Unknown benchmark: {name}. Available benchmarks: {available}"
        )
    return _BENCHMARK_REGISTRY[name_lower]


def list_benchmarks() -> list[str]:
    """List all registered benchmark names.

    Returns:
        Sorted list of benchmark names
    """
    _ensure_registry_initialized()
    return sorted(_BENCHMARK_REGISTRY.keys())
