"""Package version helpers."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import tomllib


def _read_local_pyproject_version() -> str:
    """Return the version declared in pyproject.toml for local development."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return "0.0.0"
    return data.get("project", {}).get("version", "0.0.0")


def _detect_version() -> str:
    try:
        return metadata.version("themis-eval")
    except metadata.PackageNotFoundError:  # pragma: no cover - local dev only
        return _read_local_pyproject_version()


__version__ = _detect_version()

__all__ = ["__version__"]
