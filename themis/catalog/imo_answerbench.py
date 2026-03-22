"""Catalog definition for IMO AnswerBench."""

from __future__ import annotations

from .datasets import BuiltinIMOAnswerBenchDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_math_benchmark,
    register_math,
    summarize_math,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="imo_answerbench",
    dataset_id="Hwilner/imo-answerbench",
    split="train",
    metric_id="math_equivalence",
    requires_judge=False,
    builder=build_math_benchmark,
    registrar=register_math,
    summarizer=summarize_math,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinIMOAnswerBenchDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
