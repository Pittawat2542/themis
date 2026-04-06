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

    assert len(dataset.cases) == 2
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="ZenMoore/RoleBench",
            split="test",
            files=["rolebench-eng/instruction-generalization/general/test.jsonl"],
        ),
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="ZenMoore/RoleBench",
            split="test",
            files=["rolebench-eng/instruction-generalization/role_specific/test.jsonl"],
        ),
    ]


def test_procbench_materialization_uses_combined_raw_file_source_request() -> None:
    captured: list[BenchmarkSourceRequest] = []
    benchmark = cast(BenchmarkDefinition, load("procbench:task07"))

    def loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        captured.append(request)
        return [
            {
                "problem_name": "task07_0000",
                "prompt": "Complete task 07.",
                "task_name": "task07",
                "label": {"final": "done"},
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="ifujisawa/procbench",
            split="train",
            files=["combined_dataset.jsonl"],
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
                "question_content": "Write a Python function that returns 4.",
                "public_test_cases": '[{"input": "", "output": "4\\n", "testtype": "stdin"}]',
                "platform": "leetcode",
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="livecodebench/code_generation_lite",
            split="test",
            files=["test6.jsonl"],
        )
    ]


def test_healthbench_materialization_uses_raw_file_source_request() -> None:
    captured: list[BenchmarkSourceRequest] = []
    benchmark = cast(BenchmarkDefinition, load("healthbench"))

    def loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        captured.append(request)
        return [
            {
                "prompt_id": "healthbench-1",
                "prompt": [{"role": "user", "content": "What should I do?"}],
                "rubrics": [{"criterion": "Be helpful.", "points": 2, "tags": []}],
                "example_tags": ["theme:communication"],
                "ideal_completions_data": {"ideal_completion": "Seek support."},
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="openai/healthbench",
            split="test",
            files=["2025-05-07-06-14-12_oss_eval.jsonl"],
        )
    ]


def test_superchem_materialization_uses_parquet_raw_file_source_request() -> None:
    captured: list[BenchmarkSourceRequest] = []
    benchmark = cast(BenchmarkDefinition, load("superchem:zh"))

    def loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        captured.append(request)
        return [
            {
                "uuid": "chem-1",
                "field": "chemistry",
                "question_type": "multiple_choice",
                "question_en": "What is shown?",
                "question_zh": "图中显示了什么？",
                "question_images": ["https://example.test/chem-1.png"],
                "options_en": {"A": "Alpha", "B": "Beta"},
                "options_zh": {"A": "甲", "B": "乙"},
                "answer_en": ["B"],
                "answer_zh": ["B"],
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            source_kind="huggingface_raw_files",
            dataset_id="ZehuaZhao/SUPERChem",
            split="train",
            files=["SUPERChem-500.parquet"],
        )
    ]


@pytest.mark.parametrize("benchmark_id", catalog_benchmark_ids())
def test_catalog_materializes_every_manifest_benchmark(benchmark_id: str) -> None:
    benchmark = cast(BenchmarkDefinition, load(benchmark_id))

    dataset = benchmark.materialize_dataset()

    assert dataset.cases
    assert dataset.metadata["benchmark_id"] == benchmark.benchmark_id
