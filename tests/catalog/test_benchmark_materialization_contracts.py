from __future__ import annotations

from typing import cast

import pytest

from themis.catalog import load
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.catalog.loaders import BenchmarkSourceRequest
from tests.catalog_ids import catalog_benchmark_ids


def test_rolebench_materialization_uses_raw_file_source_request() -> None:
    captured: list[BenchmarkSourceRequest] = []
    benchmark = cast(
        BenchmarkDefinition, load("rolebench:instruction_generalization_eng")
    )

    def loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        captured.append(request)
        return [
            {
                "role": "Wizard",
                "desc": "Speaks cryptically.",
                "question": "What would you say to a lost traveler?",
                "generated": ["Follow the silver river until dawn."],
                "subset": "general",
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="ZenMoore/RoleBench",
            split="test",
            files=["instruction_generalization_eng.jsonl"],
        )
    ]


def test_livecodebench_materialization_uses_raw_file_source_request() -> None:
    captured: list[BenchmarkSourceRequest] = []
    benchmark = cast(BenchmarkDefinition, load("livecodebench"))

    def loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        captured.append(request)
        return [
            {
                "question_id": "lcb-1",
                "prompt": "Write a Python function that returns 4.",
                "public_tests": [{"input": "", "output": "4\n"}],
                "language": "python",
                "execution_mode": "stdio",
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="livecodebench/code_generation_lite",
            split="test",
            files=["release_v6/test.jsonl"],
        )
    ]


@pytest.mark.parametrize("benchmark_id", catalog_benchmark_ids())
def test_catalog_materializes_every_manifest_benchmark(benchmark_id: str) -> None:
    benchmark = cast(BenchmarkDefinition, load(benchmark_id))

    dataset = benchmark.materialize_dataset()

    assert dataset.cases
    assert dataset.metadata["benchmark_id"] == benchmark.benchmark_id
