"""Catalog definition for GPQA-Diamond."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import build_mcq_benchmark, register_mcq, summarize_mcq
from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_gpqa_diamond_rows,
)


class BuiltinGPQADiamondDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_gpqa_diamond_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "question": (
                "Which option is correct?\n\n"
                "a) alpha\nb) beta\nc) gamma\nd) delta\n\n"
                "A. d\nB. a\nC. b\nD. c"
            ),
            "answer": "D",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="gpqa_diamond",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={"dataset_id": "fingertap/GPQA-Diamond", "split": "test"},
    builder=lambda definition, config: build_mcq_benchmark(
        definition,
        config,
        expected_source_field="expected",
    ),
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinGPQADiamondDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
