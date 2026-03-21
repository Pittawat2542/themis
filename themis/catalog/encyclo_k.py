"""Catalog definition for Encyclo-K."""

from __future__ import annotations

from .datasets import BuiltinEncycloKDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_mcq_benchmark,
    register_mcq,
    summarize_mcq,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="encyclo_k",
    dataset_id="m-a-p/Encyclo-K",
    split="test",
    metric_id="choice_accuracy",
    requires_judge=False,
    builder=lambda definition, config: build_mcq_benchmark(
        definition,
        config,
        expected_source_field="answer_letter",
    ),
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinEncycloKDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
