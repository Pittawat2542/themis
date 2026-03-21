"""Catalog definition for LPFQA."""

from __future__ import annotations

from .datasets import BuiltinLPFQADatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_lpfqa_benchmark,
    register_lpfqa,
    summarize_lpfqa,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="lpfqa",
    dataset_id="m-a-p/LPFQA",
    split="train",
    metric_id="lpfqa_score",
    requires_judge=True,
    builder=build_lpfqa_benchmark,
    registrar=register_lpfqa,
    summarizer=summarize_lpfqa,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinLPFQADatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
