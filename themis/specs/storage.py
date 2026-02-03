"""Storage specification for vNext workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageSpec:
    """Storage configuration for experiment persistence and caching."""

    backend: object | None = None
    path: str | Path | None = None
    cache: bool = True


__all__ = ["StorageSpec"]
