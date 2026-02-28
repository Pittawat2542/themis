"""Cache key generation utilities for deterministic storage lookups."""

import hashlib
import json
from pathlib import Path
from typing import Any

from themis.core import entities as core_entities

__all__ = ["task_cache_key", "evaluation_cache_key", "TASK_CACHE_KEY_VERSION"]

TASK_CACHE_KEY_VERSION = "k2"


def _json_default(value: Any) -> Any:
    """Best-effort stable JSON serializer for cache key fingerprinting."""
    if isinstance(value, set):
        return sorted(value, key=repr)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return repr(value)


def _stable_hash(value: Any, *, length: int = 12) -> str:
    """Return a deterministic short hash for arbitrary JSON-serializable values."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def _reference_fingerprint(task: core_entities.GenerationTask) -> str:
    reference = task.reference
    if reference is None:
        return _stable_hash(None)
    payload = {"kind": reference.kind, "value": reference.value}
    return _stable_hash(payload)


def _evaluation_config_fingerprint(evaluation_config: dict | None) -> str:
    canonical_default = {"metrics": [], "extractor": "unknown"}
    if evaluation_config:
        payload = evaluation_config
    else:
        payload = canonical_default
    return _stable_hash(payload)


def task_cache_key(task: core_entities.GenerationTask) -> str:
    """Generate cache key for a task."""
    prompt_hash = _stable_hash(
        {"template": task.prompt.spec.template, "params": task.prompt.context}
    )
    sampling = task.sampling
    sampling_key = (
        f"{sampling.temperature:.3f}-{sampling.top_p:.3f}-{sampling.max_tokens}"
    )
    model_key = task.model.model_key
    ref_hash = _reference_fingerprint(task)

    dataset_raw = task.metadata.get("dataset_id") or task.metadata.get("sample_id")
    dataset_id = str(dataset_raw) if dataset_raw is not None else ""

    manifest_hash = task.metadata.get("manifest_hash", "")
    base_key = f"{TASK_CACHE_KEY_VERSION}::{dataset_id}::{task.prompt.spec.name}::{model_key}::{sampling_key}::{prompt_hash}::{ref_hash}"

    if manifest_hash:
        return f"{base_key}::{manifest_hash}"
    return base_key


def evaluation_cache_key(
    task: core_entities.GenerationTask, evaluation_config: dict | None
) -> str:
    """Generate cache key for an evaluation result."""
    t_key = task_cache_key(task)
    config_hash = _evaluation_config_fingerprint(evaluation_config)
    return f"{t_key}::eval:{config_hash}"
