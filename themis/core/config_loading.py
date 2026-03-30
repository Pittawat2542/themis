"""Declarative experiment config loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import Field

from themis.catalog.loaders import load_symbol, load_toml, load_yaml
from themis.core.base import FrozenModel
from themis.core.security import is_secret_reference

_SINGLE_COMPONENT_FIELDS = (
    ("generation", "generator"),
    ("generation", "reducer"),
)
_LIST_COMPONENT_FIELDS = (
    ("evaluation", "metrics"),
    ("evaluation", "parsers"),
    ("evaluation", "judge_models"),
)
_STORAGE_PATH_FIELDS = ("path", "root", "blob_root")


class ExecutionComponentTargets(FrozenModel):
    generator: str
    reducer: str | None = None
    parsers: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    judge_models: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ExperimentConfigMetadata:
    config_path: Path
    base_dir: Path
    component_targets: ExecutionComponentTargets


@dataclass(frozen=True)
class LoadedExperimentConfig:
    payload: dict[str, Any]
    metadata: ExperimentConfigMetadata


def load_experiment_definition(path: str | Path, *, overrides: list[str] | None = None) -> LoadedExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    config = _load_config(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must decode to a mapping: {config_path}")

    normalized = _normalize_paths(cast(dict[str, Any], payload), base_dir=config_path.parent)
    return LoadedExperimentConfig(
        payload=_materialize_components(normalized),
        metadata=ExperimentConfigMetadata(
            config_path=config_path,
            base_dir=config_path.parent,
            component_targets=_capture_component_targets(normalized),
        ),
    )


def load_experiment_payload(path: str | Path, *, overrides: list[str] | None = None) -> dict[str, Any]:
    return load_experiment_definition(path, overrides=overrides).payload


def _load_config(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return OmegaConf.create(load_yaml(path))
    if suffix == ".toml":
        return OmegaConf.create(load_toml(path))
    raise ValueError(f"Unsupported config format: {path}")


def _normalize_paths(payload: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = dict(payload)

    storage_payload = normalized.get("storage")
    if isinstance(storage_payload, dict):
        updated_storage = dict(storage_payload)
        parameters = updated_storage.get("parameters")
        if isinstance(parameters, dict):
            updated_parameters = dict(parameters)
            for key in _STORAGE_PATH_FIELDS:
                if key in updated_parameters:
                    updated_parameters[key] = _normalize_path_value(updated_parameters[key], base_dir=base_dir)
            updated_storage["parameters"] = updated_parameters
        normalized["storage"] = updated_storage

    runtime_payload = normalized.get("runtime")
    if isinstance(runtime_payload, dict):
        updated_runtime = dict(runtime_payload)
        for key in ("queue_root", "batch_root"):
            if key in updated_runtime:
                updated_runtime[key] = _normalize_path_value(updated_runtime[key], base_dir=base_dir)
        normalized["runtime"] = updated_runtime

    return normalized


def _normalize_path_value(value: Any, *, base_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    if is_secret_reference(value):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _capture_component_targets(payload: dict[str, Any]) -> ExecutionComponentTargets:
    generation = _mapping(payload.get("generation"))
    evaluation = _mapping(payload.get("evaluation"))

    generator = _require_component_target(generation.get("generator"), field="generation.generator")
    reducer = _optional_component_target(generation.get("reducer"), field="generation.reducer")
    parsers = _component_target_list(evaluation.get("parsers"), field="evaluation.parsers")
    metrics = _component_target_list(evaluation.get("metrics"), field="evaluation.metrics")
    judge_models = _component_target_list(evaluation.get("judge_models"), field="evaluation.judge_models")

    return ExecutionComponentTargets(
        generator=generator,
        reducer=reducer,
        parsers=parsers,
        metrics=metrics,
        judge_models=judge_models,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _require_component_target(value: Any, *, field: str) -> str:
    target = _optional_component_target(value, field=field)
    if target is None:
        raise ValueError(f"Config field {field} must be a builtin component id or importable module path")
    return target


def _optional_component_target(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Config field {field} must be a builtin component id or importable module path")


def _component_target_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Config field {field} must be a list of builtin component ids or importable module paths")
    return [_require_component_target(item, field=field) for item in value]


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
