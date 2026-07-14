"""Manifest-backed catalog entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from themis.catalog.benchmarks import (
        BenchmarkCatalogEntry,
        BenchmarkDefinition,
        BenchmarkExperimentDefaults,
        BenchmarkKit,
        BenchmarkValidationCheck,
        BenchmarkValidationResult,
    )
    from themis.catalog.suites import (
        SuiteAggregation,
        SuiteDefinition,
        SuiteExpansion,
        SuiteExpansionItem,
        SuiteItem,
        SuiteRunItem,
        SuiteRunResult,
    )
    from themis import RunResult
    from themis.storage import RunStore

__all__ = [
    "builtin_component_refs",
    "BenchmarkCatalogEntry",
    "BenchmarkDefinition",
    "BenchmarkExperimentDefaults",
    "BenchmarkKit",
    "BenchmarkValidationCheck",
    "BenchmarkValidationResult",
    "build_benchmark_experiment",
    "get_benchmark_kit",
    "get_benchmark",
    "list_benchmark_ids",
    "list_benchmark_kits",
    "list_benchmarks",
    "list_component_ids",
    "load",
    "SuiteDefinition",
    "SuiteAggregation",
    "SuiteExpansion",
    "SuiteExpansionItem",
    "SuiteItem",
    "SuiteRunItem",
    "SuiteRunResult",
    "expand_suite",
    "get_suite",
    "list_suites",
    "register_suite",
    "run",
    "run_suite",
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
    overrides: BenchmarkExperimentDefaults | None = None,
    dataset=None,
):
    """Build a complete experiment from a benchmark kit."""

    from themis.catalog.benchmarks import (
        build_benchmark_experiment as _build_benchmark_experiment,
    )

    return _build_benchmark_experiment(
        name,
        overrides=overrides,
        dataset=dataset,
    )


def list_component_ids(*, kind: str | None = None) -> list[str]:
    """List builtin component identifiers, optionally filtered by kind."""

    from themis.catalog.registry import list_component_ids as _list_component_ids

    return _list_component_ids(kind=kind)


def register_suite(suite: SuiteDefinition) -> SuiteDefinition:
    """Register a suite definition for this process."""

    from themis.catalog.suites import register_suite as _register_suite

    return _register_suite(suite)


def get_suite(suite_id: str) -> SuiteDefinition:
    """Return a registered suite definition."""

    from themis.catalog.suites import get_suite as _get_suite

    return _get_suite(suite_id)


def list_suites(*, tags: list[str] | None = None) -> list[str]:
    """List registered suite identifiers."""

    from themis.catalog.suites import list_suites as _list_suites

    return _list_suites(tags=tags)


def expand_suite(suite_id: str) -> SuiteExpansion:
    """Expand a suite into benchmark-backed executable items."""

    from themis.catalog.suites import expand_suite as _expand_suite

    return _expand_suite(suite_id)


def run_suite(suite_id: str, *, store=None) -> SuiteRunResult:
    """Run all executable items in a suite."""

    from themis.catalog.suites import run_suite as _run_suite

    return _run_suite(suite_id, store=store)


def __getattr__(name: str) -> object:
    if name in {
        "BenchmarkCatalogEntry",
        "BenchmarkDefinition",
        "BenchmarkExperimentDefaults",
        "BenchmarkKit",
        "BenchmarkValidationCheck",
        "BenchmarkValidationResult",
    }:
        from themis.catalog import benchmarks

        return getattr(benchmarks, name)
    if name in {
        "SuiteAggregation",
        "SuiteDefinition",
        "SuiteExpansion",
        "SuiteExpansionItem",
        "SuiteItem",
        "SuiteRunItem",
        "SuiteRunResult",
    }:
        from themis.catalog import suites

        return getattr(suites, name)
    raise AttributeError(name)
