"""Catalog definition for BABE."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import build_mcq_benchmark, register_mcq, summarize_mcq
from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_babe_rows,
)


class BuiltinBABEDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_babe_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "uuid": "babe-1",
            "text": "This article lead is written in an opinionated voice.",
            "label": 1,
            "outlet": "Example Outlet",
            "topic": "media",
            "type": "left",
            "label_opinion": "Expresses writer's opinion",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="babe",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={"dataset_id": "mediabiasgroup/BABE", "split": "test"},
    builder=lambda definition, config: build_mcq_benchmark(
        definition,
        config,
        expected_source_field="expected",
    ),
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinBABEDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
