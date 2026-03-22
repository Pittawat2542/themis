"""Catalog definition for BeyondAIME."""

from __future__ import annotations

from themis import BenchmarkDefinition
from themis.specs.foundational import DatasetSpec

from ...common import (
    build_math_benchmark,
    register_math,
    summarize_math,
)
from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_math_short_answer_rows,
)


class BuiltinBeyondAIMEDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_math_short_answer_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "beyond-aime-1",
            "problem": "What is 7 times 6?",
            "answer": "42",
            "source": "fixture",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="beyond_aime",
    family="catalog",
    primary_metric_id="math_equivalence",
    requires_judge=False,
    metadata={"dataset_id": "ByteDance-Seed/BeyondAIME", "split": "test"},
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinBeyondAIMEDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
