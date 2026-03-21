"""Public catalog helpers and built-in benchmark catalog."""

from __future__ import annotations

from .datasets import CatalogDatasetProvider, CatalogNormalizedRows
from .runtime import (
    build_catalog_registry,
    register_catalog_engine,
    register_catalog_metrics,
)

from .common import (
    CatalogBenchmarkDefinition,
    CatalogBenchmarkRuntimeConfig,
    build_catalog_benchmark_project as _build_catalog_benchmark_project,
    inspect_huggingface_dataset,
    load_huggingface_rows,
    load_local_rows,
)
from .encyclo_k import DEFINITION as ENCYCLO_K_DEFINITION
from .healthbench import DEFINITION as HEALTHBENCH_DEFINITION
from .hle import DEFINITION as HLE_DEFINITION
from .lpfqa import DEFINITION as LPFQA_DEFINITION
from .mmlu_pro import DEFINITION as MMLU_PRO_DEFINITION
from .simpleqa_verified import DEFINITION as SIMPLEQA_VERIFIED_DEFINITION
from .supergpqa import DEFINITION as SUPERGPQA_DEFINITION

_CATALOG_BENCHMARKS: dict[str, CatalogBenchmarkDefinition] = {
    definition.benchmark_id: definition
    for definition in (
        MMLU_PRO_DEFINITION,
        SUPERGPQA_DEFINITION,
        ENCYCLO_K_DEFINITION,
        SIMPLEQA_VERIFIED_DEFINITION,
        HEALTHBENCH_DEFINITION,
        LPFQA_DEFINITION,
        HLE_DEFINITION,
    )
}


def list_catalog_benchmarks() -> list[str]:
    return sorted(_CATALOG_BENCHMARKS)


def get_catalog_benchmark(name: str) -> CatalogBenchmarkDefinition:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in _CATALOG_BENCHMARKS:
        raise ValueError(
            "Unknown built-in benchmark. Choose one of: "
            + ", ".join(sorted(_CATALOG_BENCHMARKS))
        )
    return _CATALOG_BENCHMARKS[normalized]


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
        subset=subset,
        dataset_revision=dataset_revision,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
        huggingface_loader=huggingface_loader,
    )


__all__ = [
    "CatalogBenchmarkDefinition",
    "CatalogBenchmarkRuntimeConfig",
    "CatalogDatasetProvider",
    "CatalogNormalizedRows",
    "build_catalog_benchmark_project",
    "build_catalog_registry",
    "get_catalog_benchmark",
    "inspect_huggingface_dataset",
    "list_catalog_benchmarks",
    "load_huggingface_rows",
    "load_local_rows",
    "register_catalog_engine",
    "register_catalog_metrics",
]
