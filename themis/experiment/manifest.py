"""Reproducibility manifest helpers for experiment runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from themis.exceptions import ConfigurationError

MANIFEST_SCHEMA_VERSION = "1"

_REQUIRED_FIELDS = {
    "schema_version",
    "model",
    "sampling",
    "num_samples",
    "evaluation",
    "seeds",
    "package_versions",
    "git_commit_hash",
}


def build_reproducibility_manifest(
    *,
    model: str,
    provider: str,
    provider_options: Mapping[str, Any],
    sampling: Mapping[str, Any],
    num_samples: int,
    evaluation_config: Mapping[str, Any],
    seeds: Mapping[str, Any] | None = None,
    dataset_fingerprint: str | None = None,
    prompt_fingerprint: str | None = None,
    benchmark_id: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest with deterministic structure."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": {
            "identifier": model,
            "provider": provider,
            "provider_options": dict(provider_options),
        },
        "sampling": {
            "temperature": float(sampling.get("temperature", 0.0)),
            "top_p": float(sampling.get("top_p", 0.95)),
            "max_tokens": int(sampling.get("max_tokens", 512)),
        },
        "num_samples": int(num_samples),
        "evaluation": dict(evaluation_config),
        "dataset": {
            "fingerprint": dataset_fingerprint or "unknown",
            "benchmark_id": benchmark_id,
        },
        "prompt": {
            "template_hash": prompt_fingerprint or "unknown",
        },
        "seeds": dict(seeds or {"sampling_seed": None}),
        "package_versions": _collect_package_versions(),
        "git_commit_hash": _git_commit_hash(cwd=cwd),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    validate_reproducibility_manifest(manifest)
    return manifest


def validate_reproducibility_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate manifest completeness and key scientific fields."""
    missing = sorted(field for field in _REQUIRED_FIELDS if field not in manifest)
    if missing:
        raise ConfigurationError(
            f"Missing required manifest fields: {', '.join(missing)}"
        )

    model_section = manifest.get("model")
    if not isinstance(model_section, Mapping):
        raise ConfigurationError("Manifest field 'model' must be an object.")
    for key in ("identifier", "provider", "provider_options"):
        if key not in model_section:
            raise ConfigurationError(f"Manifest field 'model.{key}' is required.")

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ConfigurationError("Manifest field 'evaluation' must be an object.")
    if "metrics" not in evaluation or "extractor" not in evaluation:
        raise ConfigurationError(
            "Manifest field 'evaluation' must include 'metrics' and 'extractor'."
        )

    commit_hash = manifest.get("git_commit_hash")
    if not isinstance(commit_hash, str) or not commit_hash.strip():
        raise ConfigurationError(
            "Manifest field 'git_commit_hash' must be a non-empty string."
        )


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Compute deterministic hash for a manifest payload."""
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_package_versions() -> dict[str, str]:
    package_names = (
        "themis-eval",
        "pydantic",
        "cyclopts",
        "litellm",
    )
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_commit_hash(*, cwd: Path | None = None) -> str:
    resolved_cwd = str(cwd) if cwd is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=resolved_cwd,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


__all__ = [
    "build_reproducibility_manifest",
    "validate_reproducibility_manifest",
    "manifest_hash",
]
