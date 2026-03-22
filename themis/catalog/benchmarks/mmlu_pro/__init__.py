"""Catalog definition for MMLU-Pro."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import (
    build_mcq_benchmark,
    register_mcq,
    summarize_mcq,
)
from ...datasets.common import BuiltinMCQDatasetProvider


class BuiltinMMLUProDatasetProvider(BuiltinMCQDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(
            metadata_keys=["category", "src"],
            huggingface_loader=huggingface_loader,
        )


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "mmlu-pro-1",
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Mercury"],
            "answer": "B",
            "answer_index": 1,
            "category": "astronomy",
            "src": "fixture",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="mmlu_pro",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={"dataset_id": "TIGER-Lab/MMLU-Pro", "split": "test"},
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
    preview_rows_loader=_preview_rows,
)
