"""Helpers for loading manifest-backed catalog entries."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import tomllib

_import_module = importlib.import_module


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_yaml(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    loaded = OmegaConf.load(path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config: {path}")
    return json.loads(json.dumps(payload))


def load_symbol(target: str) -> Any:
    module_name, _, symbol_name = target.partition(":")
    if not module_name or not symbol_name:
        raise ValueError(f"Invalid load target: {target}")
    module = _import_module(module_name)
    return getattr(module, symbol_name)
