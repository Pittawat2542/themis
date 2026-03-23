"""Catalog definition for IMO AnswerBench."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common.builders import build_math_benchmark
from ...common.registration import register_math
from ...common.summaries import summarize_math
from ...datasets._normalizers import _normalize_imo_answerbench_rows
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows


class BuiltinIMOAnswerBenchDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_imo_answerbench_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "Problem ID": "imo-bench-algebra-001",
            "Problem": "Find the value of \\frac{1}{2} + \\frac{1}{2}.",
            "Short Answer": "1",
            "Category": "Algebra",
            "Subcategory": "equations",
            "Source": "fixture",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="imo_answerbench",
    family="catalog",
    primary_metric_id="math_equivalence",
    requires_judge=False,
    metadata={"dataset_id": "Hwilner/imo-answerbench", "split": "train"},
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinIMOAnswerBenchDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
