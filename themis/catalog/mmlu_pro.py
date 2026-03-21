"""Catalog definition for MMLU-Pro."""

from __future__ import annotations

from .datasets import BuiltinMMLUProDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_mcq_benchmark,
    register_mcq,
    summarize_mcq,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="mmlu_pro",
    dataset_id="TIGER-Lab/MMLU-Pro",
    split="test",
    metric_id="choice_accuracy",
    requires_judge=False,
    builder=lambda definition, config: build_mcq_benchmark(
        definition,
        config,
        expected_source_field="answer",
    ),
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinMMLUProDatasetProvider(huggingface_loader=huggingface_loader)
    ),
)
