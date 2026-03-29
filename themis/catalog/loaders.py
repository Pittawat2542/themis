"""Helpers for loading manifest-backed catalog entries."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import tomllib

_import_module = importlib.import_module


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_symbol(target: str) -> Any:
    module_name, _, symbol_name = target.partition(":")
    if not module_name or not symbol_name:
        raise ValueError(f"Invalid load target: {target}")
    module = _import_module(module_name)
    return getattr(module, symbol_name)
