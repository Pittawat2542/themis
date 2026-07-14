"""Operational launcher loading for the CLI and deferred execution."""

from __future__ import annotations

import sys
from hashlib import sha256
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from pydantic import TypeAdapter

from themis.api import Experiment, RunOptions
from themis.catalog.loaders import load_symbol, load_toml, load_yaml
from themis.core.config import RuntimeConfig, StorageConfig
from themis.core.experiment import Experiment as CoreExperiment


def _load_runtime_experiment(
    path: str | Path, *, overrides: list[str] | None = None
) -> CoreExperiment:
    """Load a public definition into the private operational runtime model."""

    launcher_path = Path(path).expanduser().resolve()
    payload = _load_payload(launcher_path, overrides=overrides)
    definition = payload.get("definition")
    if not isinstance(definition, str) or not definition:
        raise ValueError(
            "Launcher config requires 'definition: module:symbol'; "
            "v6 does not define experiments in YAML or TOML."
        )

    loaded = _load_definition(definition, launcher_path.parent)
    if isinstance(loaded, Experiment):
        experiment = loaded
    elif callable(loaded):
        experiment = loaded()
    else:
        raise TypeError(
            f"Launcher definition {definition!r} is neither an Experiment nor callable."
        )
    if not isinstance(experiment, Experiment):
        raise TypeError(
            f"Launcher definition {definition!r} must resolve to an Experiment "
            "or a zero-argument factory returning one."
        )

    storage = StorageConfig.model_validate(payload.get("storage", {}))
    storage = _resolve_storage_paths(storage, launcher_path.parent)
    runtime_payload = dict(payload.get("runtime", {}))
    queue_root = runtime_payload.pop("queue_root", None)
    batch_root = runtime_payload.pop("batch_root", None)
    options = RunOptions.model_validate(runtime_payload)
    runtime = options._core().model_copy(
        update={
            "queue_root": _resolve_optional_path(queue_root, launcher_path.parent),
            "batch_root": _resolve_optional_path(batch_root, launcher_path.parent),
        }
    )
    core = experiment._core(storage, options)
    return core.model_copy(update={"runtime": RuntimeConfig.model_validate(runtime)})


def resolve_definition_path(path: str | Path) -> Path:
    """Resolve a launcher's local Python definition without importing it."""

    launcher_path = Path(path).expanduser().resolve()
    definition = _load_payload(launcher_path, overrides=None).get("definition")
    if not isinstance(definition, str):
        raise ValueError("Launcher config requires 'definition: module:symbol'")
    module_name, separator, symbol_name = definition.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"Invalid launcher definition: {definition}")
    definition_path = launcher_path.parent.joinpath(
        *module_name.split(".")
    ).with_suffix(".py")
    if not definition_path.is_file():
        raise ValueError(
            "Deferred execution requires a local Python executable definition"
        )
    return definition_path.resolve()


def _load_payload(path: Path, *, overrides: list[str] | None) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = load_yaml(path)
    elif path.suffix.lower() == ".toml":
        payload = load_toml(path)
    else:
        raise ValueError(f"Unsupported launcher format: {path}")
    if overrides:
        from omegaconf import OmegaConf

        merged = OmegaConf.merge(payload, OmegaConf.from_dotlist(overrides))
        payload = TypeAdapter(dict[str, Any]).validate_python(
            OmegaConf.to_container(merged, resolve=True)
        )
    return payload


def _resolve_storage_paths(config: StorageConfig, base_dir: Path) -> StorageConfig:
    kwargs = dict(config.kwargs)
    for key in ("path", "root", "blob_root"):
        value = kwargs.get(key)
        if isinstance(value, str) and not Path(value).expanduser().is_absolute():
            kwargs[key] = str((base_dir / value).resolve())
    return config.model_copy(update={"kwargs": kwargs})


def _resolve_optional_path(value: object, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _load_definition(target: str, base_dir: Path) -> object:
    module_name, separator, symbol_name = target.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"Invalid launcher definition: {target}")
    local_module = base_dir.joinpath(*module_name.split(".")).with_suffix(".py")
    if local_module.is_file():
        unique_name = (
            f"_themis_launcher_{sha256(str(local_module).encode()).hexdigest()}"
        )
        module = ModuleType(unique_name)
        module.__file__ = str(local_module)
        sys.modules[unique_name] = module
        source = local_module.read_text(encoding="utf-8")
        exec(compile(source, local_module, "exec"), module.__dict__)
        return getattr(module, symbol_name)
    with _import_path(base_dir):
        return load_symbol(target)


@contextmanager
def _import_path(path: Path) -> Iterator[None]:
    entry = str(path)
    added = entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if added:
            sys.path.remove(entry)
