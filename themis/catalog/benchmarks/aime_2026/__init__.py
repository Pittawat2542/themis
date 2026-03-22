"""Catalog definition for AIME 2026."""

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


class BuiltinMathArenaDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_math_short_answer_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "problem_idx": 1,
            "problem": "What is 2 + 2?",
            "answer": "4",
            "problem_type": ["Algebra"],
            "source": "fixture",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="aime_2026",
    family="catalog",
    primary_metric_id="math_equivalence",
    requires_judge=False,
    metadata={"dataset_id": "MathArena/aime_2026", "split": "train"},
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinMathArenaDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
