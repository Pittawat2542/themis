"""Catalog definition for SimpleQA Verified."""

from __future__ import annotations

from .datasets import BuiltinSimpleQAVerifiedDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_simpleqa_benchmark,
    register_simpleqa,
    summarize_simpleqa,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="simpleqa_verified",
    dataset_id="google/simpleqa-verified",
    split="eval",
    metric_id="simpleqa_verified_score",
    requires_judge=True,
    builder=build_simpleqa_benchmark,
    registrar=register_simpleqa,
    summarizer=summarize_simpleqa,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinSimpleQAVerifiedDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
