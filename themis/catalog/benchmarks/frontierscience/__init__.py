"""Catalog definition for FrontierScience."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import (
    build_frontierscience_benchmark,
    register_frontierscience,
    summarize_lpfqa,
)
from .dataset import BuiltinFrontierScienceDatasetProvider


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "frontierscience-1",
            "problem": "Derive the requested expression.",
            "answer": "Points: 1.0, Item: derive the expression correctly.",
            "subject": "physics",
            "task_group_id": "group-1",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="frontierscience",
    family="catalog",
    primary_metric_id="frontierscience_score",
    requires_judge=True,
    metadata={"dataset_id": "openai/frontierscience", "split": "test"},
    builder=build_frontierscience_benchmark,
    registrar=register_frontierscience,
    summarizer=summarize_lpfqa,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinFrontierScienceDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
