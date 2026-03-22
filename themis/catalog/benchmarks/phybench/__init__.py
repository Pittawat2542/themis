"""Catalog definition for PHYBench."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import build_math_benchmark, register_math, summarize_math
from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_phybench_rows,
)


class BuiltinPHYBenchDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_phybench_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "tag": "MECHANICS",
            "content": "Find the acceleration.",
            "answer": "9.8",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="phybench",
    family="catalog",
    primary_metric_id="math_equivalence",
    requires_judge=False,
    metadata={"dataset_id": "Eureka-Lab/PHYBench", "split": "train"},
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinPHYBenchDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
