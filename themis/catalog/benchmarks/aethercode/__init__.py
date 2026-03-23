"""Catalog definition for AetherCode."""

from __future__ import annotations

from themis import BenchmarkDefinition

from ...common.builders import build_aethercode_benchmark
from ...common.registration import register_aethercode
from ...common.summaries import summarize_aethercode
from .dataset import (
    DEFAULT_AETHERCODE_SUBSET,
    BuiltinAetherCodeDatasetProvider,
)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "item_id": "60173",
            "prompt_text": (
                "Write a C++17 program that solves the following problem. "
                "Return only code.\n\nProblem:\nCompute the answer."
            ),
            "language": "cpp",
            "execution_mode": "stdio",
            "official_tests": [{"input": "1\n", "output": "2\n"}],
            "generated_checker": '#include "testlib.h"\nint main() { return 0; }\n',
            "checker_language": "cpp",
            "difficulty": "Easy",
            "contest_category": "ICPC East Asia Regionals",
            "contest_name": "The 2024 ICPC Asia Shanghai Regional Contest",
            "date": "2024/11/17",
            "year": 2024,
            "metadata": {
                "difficulty": "Easy",
                "contest_category": "ICPC East Asia Regionals",
                "contest_name": "The 2024 ICPC Asia Shanghai Regional Contest",
                "date": "2024/11/17",
                "year": "2024",
            },
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="aethercode",
    family="catalog",
    primary_metric_id="aethercode_pass_rate",
    requires_judge=False,
    metadata={
        "dataset_id": "m-a-p/AetherCode",
        "split": "test",
        "subset": DEFAULT_AETHERCODE_SUBSET,
    },
    builder=build_aethercode_benchmark,
    registrar=register_aethercode,
    summarizer=summarize_aethercode,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinAetherCodeDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)
