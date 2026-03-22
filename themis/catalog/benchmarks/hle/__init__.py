"""Catalog definition for HLE."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import (
    build_hle_benchmark,
    register_hle,
    summarize_hle,
)
from .dataset import BuiltinHLEDatasetProvider


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "hle-1",
            "id": "hle-1",
            "question": "What is 12 multiplied by 12?",
            "answer": "144",
            "image": "",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="hle",
    family="catalog",
    primary_metric_id="hle_accuracy",
    requires_judge=True,
    metadata={"dataset_id": "cais/hle", "split": "test"},
    builder=build_hle_benchmark,
    registrar=register_hle,
    summarizer=summarize_hle,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinHLEDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
