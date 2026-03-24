"""Catalog definition for SimpleQA Verified."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common.builders import build_simpleqa_benchmark
from ...common.registration import register_simpleqa
from ...common.summaries import summarize_simpleqa
from .dataset import BuiltinSimpleQAVerifiedDatasetProvider


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "simpleqa-1",
            "original_index": 1,
            "problem": "What is the chemical symbol for gold?",
            "answer": "Au",
            "topic": "chemistry",
            "answer_type": "short_text",
            "multi_step": False,
            "requires_reasoning": False,
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="simpleqa_verified",
    family="catalog",
    primary_metric_id="simpleqa_verified_score",
    requires_judge=True,
    metadata={"dataset_id": "google/simpleqa-verified", "split": "eval"},
    builder=build_simpleqa_benchmark,
    registrar=register_simpleqa,
    summarizer=summarize_simpleqa,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinSimpleQAVerifiedDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
