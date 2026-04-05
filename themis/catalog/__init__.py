"""Manifest-backed catalog entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from themis.core.results import RunResult
    from themis.core.store import RunStore

__all__ = [
    "builtin_component_refs",
    "list_component_ids",
    "load",
    "run",
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


def list_component_ids(*, kind: str | None = None) -> list[str]:
    """List builtin component identifiers, optionally filtered by kind."""

    from themis.catalog.registry import list_component_ids as _list_component_ids

    return _list_component_ids(kind=kind)
