"""Catalog definition for BeyondAIME."""

from __future__ import annotations

from .datasets import BuiltinBeyondAIMEDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_math_benchmark,
    register_math,
    summarize_math,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="beyond_aime",
    dataset_id="ByteDance-Seed/BeyondAIME",
    split="test",
    metric_id="math_equivalence",
    requires_judge=False,
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinBeyondAIMEDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
