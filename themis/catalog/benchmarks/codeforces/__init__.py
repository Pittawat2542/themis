"""Catalog definition for Codeforces."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common import (
    build_codeforces_benchmark,
    register_codeforces,
    summarize_codeforces,
)
from .dataset import BuiltinOpenR1CodeforcesDatasetProvider


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "id": "1A",
            "contest_id": "1",
            "title": "Theatre Square",
            "description": "Compute the number of flagstones needed.",
            "input_format": "Three integers n, m, a.",
            "output_format": "Print one integer.",
            "interaction_format": None,
            "time_limit": 1.0,
            "memory_limit": 256.0,
            "official_tests_complete": True,
            "official_tests": [{"input": "6 6 4\n", "output": "4\n"}],
            "input_mode": "stdio",
            "generated_checker": None,
            "executable": True,
            "generated_tests": 0,
            "language": "python",
            "prompt": "Write a Python program that solves the problem.",
            "rating": 1000,
            "tags": ["math"],
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="codeforces",
    family="catalog",
    primary_metric_id="codeforces_pass_rate",
    requires_judge=False,
    metadata={
        "dataset_id": "open-r1/codeforces",
        "split": "test",
        "subset": "verifiable-prompts",
    },
    builder=build_codeforces_benchmark,
    registrar=register_codeforces,
    summarizer=summarize_codeforces,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinOpenR1CodeforcesDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
