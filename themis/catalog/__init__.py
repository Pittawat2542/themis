"""Public catalog helpers and built-in benchmark catalog."""

from __future__ import annotations

from themis import BenchmarkDefinition, BenchmarkDefinitionConfig

from . import metrics
from .benchmarks import get_catalog_benchmark, list_catalog_benchmarks
from .datasets import CatalogDatasetProvider, CatalogNormalizedRows
from .runtime import (
    build_catalog_registry,
    register_catalog_engine,
    register_catalog_metrics,
)

from .common import (
    build_catalog_benchmark_project as _build_catalog_benchmark_project,
    inspect_huggingface_dataset,
    load_huggingface_rows,
    load_local_rows,
)


def build_catalog_benchmark_project(
    *,
    benchmark_id: str,
    model_id: str,
    provider: str,
    storage_root,
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
):
    return _build_catalog_benchmark_project(
        benchmark_id=benchmark_id,
        model_id=model_id,
        provider=provider,
        storage_root=storage_root,
        get_catalog_benchmark=get_catalog_benchmark,
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


__all__ = [
    "BenchmarkDefinition",
    "BenchmarkDefinitionConfig",
    "CatalogDatasetProvider",
    "CatalogNormalizedRows",
    "build_catalog_benchmark_project",
    "build_catalog_registry",
    "get_catalog_benchmark",
    "inspect_huggingface_dataset",
    "list_catalog_benchmarks",
    "load_huggingface_rows",
    "load_local_rows",
    "metrics",
    "register_catalog_engine",
    "register_catalog_metrics",
]
