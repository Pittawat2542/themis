"""Declarative experiment config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from themis.catalog.loaders import load_symbol, load_toml, load_yaml

_SINGLE_COMPONENT_FIELDS = (
    ("generation", "generator"),
    ("generation", "reducer"),
)
_LIST_COMPONENT_FIELDS = (
    ("evaluation", "metrics"),
    ("evaluation", "parsers"),
    ("evaluation", "judge_models"),
)


def load_experiment_payload(path: str | Path, *, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    config = _load_config(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must decode to a mapping: {config_path}")
    return _materialize_components(cast(dict[str, Any], payload))


def _load_config(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return OmegaConf.create(load_yaml(path))
    if suffix == ".toml":
        return OmegaConf.create(load_toml(path))
    raise ValueError(f"Unsupported config format: {path}")


def _materialize_components(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for group, field in _SINGLE_COMPONENT_FIELDS:
        group_payload = normalized.get(group)
        if isinstance(group_payload, dict) and field in group_payload:
            updated_group = dict(group_payload)
            updated_group[field] = _maybe_load_component(updated_group[field])
            normalized[group] = updated_group

    for group, field in _LIST_COMPONENT_FIELDS:
        group_payload = normalized.get(group)
        if isinstance(group_payload, dict) and isinstance(group_payload.get(field), list):
            updated_group = dict(group_payload)
            updated_group[field] = [_maybe_load_component(item) for item in updated_group[field]]
            normalized[group] = updated_group

    return normalized


def _maybe_load_component(value: Any) -> Any:
    if not isinstance(value, str) or ":" not in value:
        return value
    loaded = load_symbol(value)
    if isinstance(loaded, type):
        return loaded()
    return loaded
