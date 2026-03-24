"""Catalog definition for Encyclo-K."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common.builders import build_mcq_benchmark
from ...common.registration import register_mcq
from ...common.summaries import summarize_mcq
from ...datasets._providers import BuiltinMCQDatasetProvider


class BuiltinEncycloKDatasetProvider(BuiltinMCQDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(
            metadata_keys=["discipline", "field", "subfield", "difficulty"],
            huggingface_loader=huggingface_loader,
        )


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "encyclo-k-1",
            "question": "What is the capital city of Canada?",
            "options": ["Toronto", "Vancouver", "Ottawa", "Montreal"],
            "answer_letter": "C",
            "discipline": "humanities",
            "field": "geography",
            "subfield": "capitals",
            "difficulty": "easy",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="encyclo_k",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={"dataset_id": "m-a-p/Encyclo-K", "split": "test"},
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
    preview_rows_loader=_preview_rows,
)
