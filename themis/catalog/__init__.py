"""Manifest-backed catalog entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from themis.catalog.benchmarks import (
        BenchmarkCatalogEntry,
        BenchmarkExperimentDefaults,
        BenchmarkKit,
        BenchmarkValidationResult,
    )
    from themis.core.results import RunResult
    from themis.core.store import RunStore

__all__ = [
    "builtin_component_refs",
    "build_benchmark_experiment",
    "get_benchmark_kit",
    "get_benchmark",
    "list_benchmark_ids",
    "list_benchmark_kits",
    "list_benchmarks",
    "list_component_ids",
    "load",
    "run",
    "validate_benchmark",
]


def load(name: str) -> object:
    """Load a builtin component or named benchmark from the shipped catalog."""

    from themis.catalog.benchmarks import load_benchmark
    from themis.catalog.registry import load as load_component

    try:
        return load_component(name)
    except ValueError:
        return load_benchmark(name)


def run(
    name: str, *, model: object | None = None, store: RunStore | None = None
) -> RunResult:
    """Execute a named benchmark through the catalog convenience layer."""

    from themis.catalog.benchmarks import run_benchmark

    return run_benchmark(name, model=model, store=store)


def builtin_component_refs() -> dict[str, Any]:
    """Return component references for the builtin shipped catalog entries."""

    from themis.catalog.registry import (
        builtin_component_refs as _builtin_component_refs,
    )

    return _builtin_component_refs()


def list_benchmark_ids() -> list[str]:
    """List canonical benchmark identifiers from the shipped catalog."""

    from themis.catalog.benchmarks import list_benchmark_ids as _list_benchmark_ids

    return _list_benchmark_ids()


def list_benchmarks() -> list[BenchmarkCatalogEntry]:
    """Return structured metadata for shipped catalog benchmarks."""

    from themis.catalog.benchmarks import list_benchmarks as _list_benchmarks

    return _list_benchmarks()


def get_benchmark(name: str) -> BenchmarkCatalogEntry:
    """Return structured metadata for a shipped catalog benchmark."""

    from themis.catalog.benchmarks import get_benchmark as _get_benchmark

    return _get_benchmark(name)


def validate_benchmark(name: str) -> BenchmarkValidationResult:
    """Validate that a shipped benchmark can load, materialize, and score."""

    from themis.catalog.benchmarks import validate_benchmark as _validate_benchmark

    return _validate_benchmark(name)


def list_benchmark_kits() -> list[str]:
    """List benchmark identifiers that can build complete experiments."""

    from themis.catalog.benchmarks import list_benchmark_kits as _list_benchmark_kits

    return _list_benchmark_kits()


def get_benchmark_kit(name: str) -> BenchmarkKit:
    """Return a Themis-owned experiment-building kit for a benchmark."""

    from themis.catalog.benchmarks import get_benchmark_kit as _get_benchmark_kit

    return _get_benchmark_kit(name)


def build_benchmark_experiment(
    name: str,
    *,
    storage=None,
    runtime=None,
    overrides: BenchmarkExperimentDefaults | None = None,
    dataset=None,
):
    """Build a complete experiment from a benchmark kit."""

    from themis.catalog.benchmarks import (
        build_benchmark_experiment as _build_benchmark_experiment,
    )

    return _build_benchmark_experiment(
        name,
        storage=storage,
        runtime=runtime,
        overrides=overrides,
        dataset=dataset,
    )


def list_component_ids(*, kind: str | None = None) -> list[str]:
    """List builtin component identifiers, optionally filtered by kind."""

    from themis.catalog.registry import list_component_ids as _list_component_ids

    return _list_component_ids(kind=kind)
