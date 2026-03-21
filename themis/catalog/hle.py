"""Catalog definition for HLE."""

from __future__ import annotations

from .datasets import BuiltinHLEDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_hle_benchmark,
    register_hle,
    summarize_hle,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="hle",
    dataset_id="cais/hle",
    split="test",
    metric_id="hle_accuracy",
    requires_judge=True,
    builder=build_hle_benchmark,
    registrar=register_hle,
    summarizer=summarize_hle,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinHLEDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
