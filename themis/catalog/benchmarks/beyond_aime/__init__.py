"""Catalog definition for BeyondAIME."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common.builders import build_math_benchmark
from ...common.registration import register_math
from ...common.summaries import summarize_math
from ...datasets._normalizers import _normalize_math_short_answer_rows
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows


class BuiltinBeyondAIMEDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
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
