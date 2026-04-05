from __future__ import annotations

from typing import cast

from themis.catalog import load, run
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.core.base import JSONValue
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore
from tests.catalog_ids import catalog_benchmark_ids


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
    assert isinstance(task, str)
    assert task.startswith("Complete procbench task07")
    assert experiment.datasets[0].metadata["benchmark_id"] == "procbench:task07"


def test_superchem_and_mmmlu_variants_propagate_language_metadata() -> None:
    superchem = cast(BenchmarkDefinition, load("superchem:zh"))
    mmmlu = cast(BenchmarkDefinition, load("mmmlu:thai"))
    superchem_input = _sample_input(superchem)
    mmmlu_input = _sample_input(mmmlu)

    assert superchem.sample_case_metadata["language"] == "zh"
    assert superchem_input["language"] == "zh"
    assert mmmlu.sample_case_metadata["language_config"] == "thai"
    assert mmmlu_input["language"] == "thai"


def test_hle_and_humaneval_variants_preserve_variant_shapes() -> None:
    hle = cast(BenchmarkDefinition, load("hle:math,reasoning"))
    humaneval = cast(BenchmarkDefinition, load("humaneval:v0.1.0"))
    humaneval_plus = cast(BenchmarkDefinition, load("humaneval_plus:noextreme"))

    assert hle.metric_ids == ["builtin/panel_of_judges"]
    assert hle.sample_case_metadata["domains"] == "math,reasoning"
    assert humaneval.requires_code_execution is True
    assert humaneval.sample_case_metadata["variant"] == "v0.1.0"
    assert humaneval_plus.sample_case_metadata["variant"] == "noextreme"


def test_catalog_load_covers_all_benchmark_entries_from_catalog_md() -> None:
    loaded = [
        cast(BenchmarkDefinition, load(benchmark_id))
        for benchmark_id in catalog_benchmark_ids()
    ]

    assert [benchmark.benchmark_id for benchmark in loaded] == catalog_benchmark_ids()


def test_catalog_run_executes_variant_backed_benchmarks() -> None:
    rolebench_store = InMemoryRunStore()
    procbench_store = InMemoryRunStore()
    hle_store = InMemoryRunStore()

    rolebench_result = run("rolebench:role_generalization_eng", store=rolebench_store)
    procbench_result = run("procbench:task03", store=procbench_store)
    hle_result = run("hle:math,reasoning", store=hle_store)

    assert rolebench_result.status is RunStatus.COMPLETED
    assert procbench_result.status is RunStatus.COMPLETED
    assert hle_result.status is RunStatus.COMPLETED
