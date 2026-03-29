"""Manifest-backed catalog entry points."""

from __future__ import annotations

from themis.catalog.registry import builtin_component_refs, list_component_ids, load

__all__ = [
    "builtin_component_refs",
    "list_component_ids",
    "load",
]
