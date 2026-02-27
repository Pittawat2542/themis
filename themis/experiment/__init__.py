"""Experiment orchestration layer."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy-load submodules to avoid circular imports with config."""
    import importlib

    _LAZY = {"export", "math", "orchestrator"}
    if name in _LAZY:
        return importlib.import_module(f"themis.experiment.{name}")
    raise AttributeError(f"module 'themis.experiment' has no attribute {name!r}")


__all__ = ["math", "orchestrator", "export"]
