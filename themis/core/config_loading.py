"""Declarative experiment config loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import Field, model_validator

from themis.catalog.loaders import load_toml, load_yaml
from themis.core.base import FrozenModel
from themis.core.config import TargetSpec
from themis.core.security import is_secret_reference

_SINGLE_COMPONENT_FIELDS = (
    ("generation", "generator"),
    ("generation", "selector"),
    ("generation", "reducer"),
)
_LIST_COMPONENT_FIELDS = (
    ("evaluation", "metrics"),
    ("evaluation", "parsers"),
    ("evaluation", "judge_models"),
)
_STORAGE_PATH_FIELDS = ("path", "root", "blob_root")


class ExecutionComponentTargets(FrozenModel):
    generator: TargetSpec
    selector: TargetSpec | None = None
    reducer: TargetSpec | None = None
    parsers: list[TargetSpec] = Field(default_factory=list)
    metrics: list[TargetSpec] = Field(default_factory=list)
    judge_models: list[TargetSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_strings(cls, payload):
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        for key in ("generator", "selector", "reducer"):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = TargetSpec(target=value)
        for key in ("parsers", "metrics", "judge_models"):
            value = normalized.get(key)
            if isinstance(value, list):
                normalized[key] = [
                    TargetSpec(target=item) if isinstance(item, str) else item
                    for item in value
                ]
        return normalized


@dataclass(frozen=True)
class ExperimentConfigMetadata:
    config_path: Path
    base_dir: Path
    component_targets: ExecutionComponentTargets


@dataclass(frozen=True)
class LoadedExperimentConfig:
    payload: dict[str, Any]
    metadata: ExperimentConfigMetadata


def load_experiment_definition(
    path: str | Path, *, overrides: list[str] | None = None
) -> LoadedExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    config = _load_config(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must decode to a mapping: {config_path}")

    normalized = _normalize_paths(
        cast(dict[str, Any], payload), base_dir=config_path.parent
    )
    normalized = _normalize_component_specs(normalized)
    return LoadedExperimentConfig(
        payload=normalized,
        metadata=ExperimentConfigMetadata(
            config_path=config_path,
            base_dir=config_path.parent,
            component_targets=_capture_component_targets(normalized),
        ),
    )


def load_experiment_payload(
    path: str | Path, *, overrides: list[str] | None = None
) -> dict[str, Any]:
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
        if "store" in updated_storage:
            updated_storage["target"] = updated_storage.pop("store")
        if "parameters" in updated_storage and "kwargs" not in updated_storage:
            updated_storage["kwargs"] = updated_storage.pop("parameters")
        parameters = updated_storage.get("kwargs")
        if isinstance(parameters, dict):
            updated_parameters = dict(parameters)
            for key in _STORAGE_PATH_FIELDS:
                if key in updated_parameters:
                    updated_parameters[key] = _normalize_path_value(
                        updated_parameters[key], base_dir=base_dir
                    )
            updated_storage["kwargs"] = updated_parameters
        normalized["storage"] = updated_storage

    runtime_payload = normalized.get("runtime")
    if isinstance(runtime_payload, dict):
        updated_runtime = dict(runtime_payload)
        for key in ("queue_root", "batch_root"):
            if key in updated_runtime:
                updated_runtime[key] = _normalize_path_value(
                    updated_runtime[key], base_dir=base_dir
                )
        normalized["runtime"] = updated_runtime

    return normalized


def _normalize_component_specs(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)

    generation = normalized.get("generation")
    if isinstance(generation, dict):
        updated_generation = dict(generation)
        for field in ("generator", "selector", "reducer"):
            if field in updated_generation and updated_generation[field] is not None:
                updated_generation[field] = _require_component_target(
                    updated_generation[field],
                    field=f"generation.{field}",
                )
        normalized["generation"] = updated_generation

    evaluation = normalized.get("evaluation")
    if isinstance(evaluation, dict):
        updated_evaluation = dict(evaluation)
        for field in ("metrics", "parsers", "judge_models"):
            if field in updated_evaluation:
                updated_evaluation[field] = _component_target_list(
                    updated_evaluation[field],
                    field=f"evaluation.{field}",
                )
        normalized["evaluation"] = updated_evaluation

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

    generator = _require_component_target(
        generation.get("generator"), field="generation.generator"
    )
    selector = _optional_component_target(
        generation.get("selector"), field="generation.selector"
    )
    reducer = _optional_component_target(
        generation.get("reducer"), field="generation.reducer"
    )
    parsers = _component_target_list(
        evaluation.get("parsers"), field="evaluation.parsers"
    )
    metrics = _component_target_list(
        evaluation.get("metrics"), field="evaluation.metrics"
    )
    judge_models = _component_target_list(
        evaluation.get("judge_models"), field="evaluation.judge_models"
    )

    return ExecutionComponentTargets(
        generator=generator,
        selector=selector,
        reducer=reducer,
        parsers=parsers,
        metrics=metrics,
        judge_models=judge_models,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _require_component_target(value: Any, *, field: str) -> TargetSpec:
    target = _optional_component_target(value, field=field)
    if target is None:
        raise ValueError(
            f"Config field {field} must be a builtin component id, importable module path, or target+kwargs mapping"
        )
    return target


def _optional_component_target(value: Any, *, field: str) -> TargetSpec | None:
    if value is None:
        return None
    if isinstance(value, TargetSpec):
        return value
    if isinstance(value, str):
        return TargetSpec(target=value)
    if isinstance(value, dict):
        target = value.get("target")
        if not isinstance(target, str):
            raise ValueError(f"Config field {field} mapping must include string target")
        kwargs = value.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(f"Config field {field} mapping kwargs must be a mapping")
        return TargetSpec(target=target, kwargs=dict(kwargs))
    raise ValueError(
        f"Config field {field} must be a builtin component id, importable module path, or target+kwargs mapping"
    )


def _component_target_list(value: Any, *, field: str) -> list[TargetSpec]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"Config field {field} must be a list of builtin component ids, importable module paths, or target+kwargs mappings"
        )
    return [_require_component_target(item, field=field) for item in value]
