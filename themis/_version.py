"""Package version helpers."""

from __future__ import annotations

from importlib import metadata


def _detect_version() -> str:
    try:
        return metadata.version("themis")
    except metadata.PackageNotFoundError:  # pragma: no cover - local dev only
        return "0.0.0"


__version__ = _detect_version()

__all__ = ["__version__"]
