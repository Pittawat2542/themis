"""Catalog definition for HealthBench."""

from __future__ import annotations

from .datasets import BuiltinHealthBenchDatasetProvider

from .common import (
    CatalogBenchmarkDefinition,
    build_healthbench_benchmark,
    register_healthbench,
    render_healthbench_preview,
    summarize_healthbench,
)

DEFINITION = CatalogBenchmarkDefinition(
    benchmark_id="healthbench",
    dataset_id="openai/healthbench",
    split="test",
    metric_id="healthbench_score",
    requires_judge=True,
    builder=build_healthbench_benchmark,
    registrar=register_healthbench,
    summarizer=summarize_healthbench,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinHealthBenchDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_renderer=render_healthbench_preview,
)
