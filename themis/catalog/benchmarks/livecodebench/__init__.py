"""Catalog definition for LiveCodeBench code generation."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common.builders import build_livecodebench_benchmark
from ...common.registration import register_livecodebench
from ...common.summaries import summarize_livecodebench
from .dataset import (
    DEFAULT_LIVECODEBENCH_VERSION_TAG,
    BuiltinLiveCodeBenchDatasetProvider,
)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "abc387_b",
            "prompt_text": (
                "Write a Python 3 program that solves the following problem. "
                "Return only code.\n\nProblem:\nRead X and print the requested sum."
            ),
            "language": "python",
            "execution_mode": "stdio",
            "official_tests": [{"input": "1", "output": "2024\n"}],
            "metadata": {
                "platform": "atcoder",
                "contest_id": "abc387",
                "difficulty": "easy",
                "contest_date": "2025-01-04T00:00:00",
            },
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="livecodebench",
    family="catalog",
    primary_metric_id="livecodebench_pass_rate",
    requires_judge=False,
    metadata={
        "dataset_id": "livecodebench/code_generation_lite",
        "split": "test",
        "version_tag": DEFAULT_LIVECODEBENCH_VERSION_TAG,
    },
    builder=build_livecodebench_benchmark,
    registrar=register_livecodebench,
    summarizer=summarize_livecodebench,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinLiveCodeBenchDatasetProvider(
            version_tag=DEFAULT_LIVECODEBENCH_VERSION_TAG,
            huggingface_loader=huggingface_loader,
        )
    ),
    preview_rows_loader=_preview_rows,
)
