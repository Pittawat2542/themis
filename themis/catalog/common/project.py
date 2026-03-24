"""Catalog project construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from themis import (
    BenchmarkDefinition,
    BenchmarkSpec,
    PluginRegistry,
    ProjectSpec,
    build_benchmark_definition_project,
)
from themis.contracts.protocols import DatasetProvider

from .. import runtime as _runtime


def build_catalog_benchmark_project(
    *,
    benchmark_id: str,
    model_id: str,
    provider: str,
    storage_root: Path,
    get_catalog_benchmark: Callable[[str], BenchmarkDefinition],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = None,
    num_samples: int = 1,
    subset: int | None = None,
    dataset_revision: str | None = None,
    judge_model_id: str | None = None,
    judge_provider: str | None = None,
    huggingface_loader=None,
) -> tuple[
    ProjectSpec,
    BenchmarkSpec,
    PluginRegistry,
    DatasetProvider,
    BenchmarkDefinition,
]:
    definition = get_catalog_benchmark(benchmark_id)
    return build_benchmark_definition_project(
        definition=definition,
        model_id=model_id,
        provider=provider,
        storage_root=storage_root,
        build_registry=_runtime.build_catalog_registry,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        num_samples=num_samples,
        subset=subset,
        dataset_revision=dataset_revision,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
        huggingface_loader=huggingface_loader,
    )
