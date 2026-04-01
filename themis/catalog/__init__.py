"""Manifest-backed catalog entry points."""

from __future__ import annotations

__all__ = [
    "builtin_component_refs",
    "list_component_ids",
    "load",
    "run",
]


def load(name: str):
    from themis.catalog.benchmarks import load_benchmark
    from themis.catalog.registry import load as load_component

    try:
        return load_component(name)
    except ValueError:
        return load_benchmark(name)


def run(name: str, *, model: object | None = None, store=None):
    from themis.catalog.benchmarks import run_benchmark

    return run_benchmark(name, model=model, store=store)


def builtin_component_refs():
    from themis.catalog.registry import (
        builtin_component_refs as _builtin_component_refs,
    )

    return _builtin_component_refs()


def list_component_ids(*, kind: str | None = None):
    from themis.catalog.registry import list_component_ids as _list_component_ids

    return _list_component_ids(kind=kind)
