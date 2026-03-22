"""Catalog definition for LPFQA."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import (
    build_lpfqa_benchmark,
    register_lpfqa,
    summarize_lpfqa,
)
from .dataset import BuiltinLPFQADatasetProvider


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "lpfqa-1",
            "prompt_id": "lpfqa-1",
            "prompt": "Translate the phrase 'good morning' into Japanese.",
            "response_reference": "<参考答案>: おはようございます\n<评估要点>: The answer should convey a polite morning greeting.",
            "judge_prompt_template": "Reference:\n{response_reference}\n\nResponse:\n{response}",
            "judge_system_prompt": "You are a careful grading assistant.",
            "primary_domain": "translation",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="lpfqa",
    family="catalog",
    primary_metric_id="lpfqa_score",
    requires_judge=True,
    metadata={"dataset_id": "m-a-p/LPFQA", "split": "train"},
    builder=build_lpfqa_benchmark,
    registrar=register_lpfqa,
    summarizer=summarize_lpfqa,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinLPFQADatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
