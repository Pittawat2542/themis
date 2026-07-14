from __future__ import annotations

from typing import cast

import pytest

from themis.catalog import load, run
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.core.base import JSONValue
from themis.core.dataset_sources import DatasetSourceSpec
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore


def _sample_input(benchmark: BenchmarkDefinition) -> dict[str, JSONValue]:
    sample_input = benchmark.sample_case_input
    assert isinstance(sample_input, dict)
    return sample_input


def test_rolebench_variant_uses_variant_specific_rubric() -> None:
    benchmark = cast(
        BenchmarkDefinition, load("rolebench:instruction_generalization_eng")
    )
    experiment = benchmark.build_experiment()
    rubric = benchmark.workflow_overrides["rubric"]

    assert benchmark.metric_ids == ["builtin/llm_rubric"]
    assert isinstance(rubric, str)
    assert rubric.startswith(
        "Judge whether the response follows the requested role behavior"
    )
    assert benchmark.sample_case_metadata["variant"] == "instruction_generalization_eng"
    assert experiment.evaluation.metrics == ["builtin/llm_rubric"]


def test_procbench_variant_exposes_task_specific_metadata() -> None:
    benchmark = cast(BenchmarkDefinition, load("procbench:task07"))
    experiment = benchmark.build_experiment()
    sample_input = _sample_input(benchmark)

    assert benchmark.metric_ids == ["builtin/llm_rubric"]
    assert benchmark.sample_case_metadata["task_id"] == "task07"
    task = sample_input["task"]
    source = experiment.datasets[0]
    assert isinstance(task, str)
    assert isinstance(source, DatasetSourceSpec)
    assert task.startswith("Complete procbench task07")
    assert source.provenance_metadata["benchmark_id"] == "procbench:task07"


def test_superchem_and_mmmlu_variants_propagate_language_metadata() -> None:
    superchem = cast(BenchmarkDefinition, load("superchem:zh"))
    mmmlu = cast(BenchmarkDefinition, load("mmmlu:ZH_CN"))
    superchem_input = _sample_input(superchem)
    mmmlu_input = _sample_input(mmmlu)

    assert superchem.sample_case_metadata["language"] == "zh"
    assert superchem_input["language"] == "zh"
    assert mmmlu.sample_case_metadata["language_config"] == "ZH_CN"
    assert mmmlu_input["language"] == "ZH_CN"


def test_superchem_materializes_mcq_prompts_with_more_than_ten_options() -> None:
    benchmark = cast(BenchmarkDefinition, load("superchem"))

    def loader(_request) -> list[dict[str, object]]:
        return [
            {
                "uuid": "chem-many-options",
                "field": "chemistry",
                "question_type": "multiple_choice",
                "question_en": "Which option is correct?",
                "question_images": [],
                "options_en": {
                    "A": "option 1",
                    "B": "option 2",
                    "C": "option 3",
                    "D": "option 4",
                    "E": "option 5",
                    "F": "option 6",
                    "G": "option 7",
                    "H": "option 8",
                    "I": "option 9",
                    "J": "option 10",
                    "K": "option 11",
                },
                "answer_en": ["K"],
            }
        ]

    dataset = benchmark.materialize_dataset(loader=loader)

    assert len(dataset.cases) == 1
    prompt = dataset.cases[0].input
    assert isinstance(prompt, str)
    assert "K. option 11" in prompt
    assert dataset.cases[0].expected_output == {"choice": "K"}


def test_hle_variant_and_humaneval_plus_preserve_variant_shapes() -> None:
    hle = cast(BenchmarkDefinition, load("hle:math,reasoning"))
    humaneval_plus = cast(BenchmarkDefinition, load("humaneval_plus"))

    assert hle.metric_ids == ["builtin/panel_of_judges"]
    assert hle.sample_case_metadata["domains"] == "math,reasoning"
    assert humaneval_plus.requires_code_execution is True
    assert humaneval_plus.sample_case_metadata["variant"] == ""


@pytest.mark.slow
def test_catalog_runs_open_procbench_variant(
    catalog_fixture_loader: None,
) -> None:
    procbench_store = InMemoryRunStore()

    procbench_result = run("procbench:task03", store=procbench_store)

    assert procbench_result.status is RunStatus.COMPLETED
