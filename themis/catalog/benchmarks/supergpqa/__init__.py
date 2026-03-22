"""Catalog definition for SuperGPQA."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import (
    build_mcq_benchmark,
    register_mcq,
    summarize_mcq,
)
from ...datasets.common import BuiltinMCQDatasetProvider


class BuiltinSuperGPQADatasetProvider(BuiltinMCQDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(
            metadata_keys=["discipline", "field", "subfield", "difficulty"],
            huggingface_loader=huggingface_loader,
        )


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "supergpqa-1",
            "question": "Which gas is most abundant in Earth's atmosphere?",
            "options": ["Oxygen", "Hydrogen", "Nitrogen", "Carbon dioxide"],
            "answer_letter": "C",
            "discipline": "science",
            "field": "chemistry",
            "subfield": "atmospheric chemistry",
            "difficulty": "medium",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="supergpqa",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={"dataset_id": "m-a-p/SuperGPQA", "split": "train"},
    builder=lambda definition, config: build_mcq_benchmark(
        definition,
        config,
        expected_source_field="answer_letter",
    ),
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinSuperGPQADatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
