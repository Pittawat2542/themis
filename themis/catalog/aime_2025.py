"""Catalog definition for AIME 2025."""

from __future__ import annotations

from .datasets import BuiltinMathArenaDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_math_benchmark,
    register_math,
    summarize_math,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="aime_2025",
    dataset_id="MathArena/aime_2025",
    split="train",
    metric_id="math_equivalence",
    requires_judge=False,
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinMathArenaDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
