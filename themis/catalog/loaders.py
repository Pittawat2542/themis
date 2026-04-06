"""Helpers for loading manifest-backed catalog entries."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import tomllib
from pydantic import Field

from themis.core.base import FrozenModel

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


class MissingOptionalDependencyError(RuntimeError):
    """Raised when an optional dependency is unavailable."""


class BenchmarkSourceRequest(FrozenModel):
    source_kind: str = "huggingface_dataset"
    dataset_id: str
    split: str
    revision: str | None = None
    config_name: str | None = None
    files: list[str] = Field(default_factory=list)


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    *,
    config_name: str | None = None,
) -> list[dict[str, object]]:
    try:
        datasets_module = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError(
            "Catalog dataset loading requires the optional datasets dependency. "
            'Install it with: uv add "themis-eval[datasets]"'
        ) from exc

    args: list[object] = [dataset_id]
    if config_name is not None:
        args.append(config_name)
    dataset = datasets_module.load_dataset(
        *args,
        split=split,
        revision=revision,
    )
    return [dict(row) for row in dataset]


def load_huggingface_raw_rows(
    dataset_id: str,
    *,
    files: list[str],
    revision: str | None = None,
) -> list[dict[str, object]]:
    if not files:
        raise ValueError(f"No raw files configured for dataset {dataset_id}")
    try:
        hub_module = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError(
            "Catalog raw benchmark loading requires the optional "
            "`huggingface_hub` module. Install it with: "
            "pip install huggingface-hub or "
            'uv add "themis-eval[datasets]"'
        ) from exc

    rows: list[dict[str, object]] = []
    for filename in files:
        local_path = hub_module.hf_hub_download(
            repo_id=dataset_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
        rows.extend(_read_jsonl_rows(Path(local_path)))
    return rows


def load_benchmark_rows(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
    if request.source_kind == "huggingface_dataset":
        return load_huggingface_rows(
            request.dataset_id,
            request.split,
            request.revision,
            config_name=request.config_name,
        )
    if request.source_kind == "huggingface_raw_files":
        return load_huggingface_raw_rows(
            request.dataset_id,
            files=request.files,
            revision=request.revision,
        )
    raise ValueError(f"Unknown benchmark source kind: {request.source_kind}")


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append({str(key): value for key, value in parsed.items()})
    return rows
