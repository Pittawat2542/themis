from __future__ import annotations

from typing import cast

from themis.catalog import load, run
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.core.base import JSONValue
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore


def _sample_input(benchmark: BenchmarkDefinition) -> dict[str, JSONValue]:
    sample_input = benchmark.sample_case_input
    assert isinstance(sample_input, dict)
    return sample_input


def test_rubric_benchmark_adapter_configures_workflow_metrics() -> None:
    benchmark = cast(BenchmarkDefinition, load("frontierscience"))
    experiment = benchmark.build_experiment()

    assert benchmark.metric_ids == ["builtin/llm_rubric"]
    assert benchmark.judge_model_ids == ["builtin/demo_judge"]
    assert (
        benchmark.workflow_overrides["rubric"]
        == "Judge whether the scientific answer is correct and well-supported."
    )
    assert experiment.evaluation.metrics == ["builtin/llm_rubric"]
    assert experiment.evaluation.judge_models == ["builtin/demo_judge"]


def test_code_benchmark_adapter_configures_best_of_n_and_execution_metadata() -> None:
    benchmark = cast(BenchmarkDefinition, load("codeforces"))
    experiment = benchmark.build_experiment()
    sample_input = _sample_input(benchmark)

    assert benchmark.selector_id == "builtin/best_of_n"
    assert benchmark.judge_model_ids == ["builtin/demo_judge"]
    assert benchmark.candidate_policy == {"num_samples": 2}
    problem = sample_input["problem"]
    assert isinstance(problem, str)
    assert problem.startswith("Solve the programming task")
    assert experiment.generation.selector == "builtin/best_of_n"
    assert experiment.generation.reducer is None
    assert (
        experiment.datasets[0].metadata["supported_execution_backends"]
        == "piston,sandbox_fusion"
    )


def test_catalog_run_executes_representative_adapter_backed_benchmarks() -> None:
    frontierscience_store = InMemoryRunStore()
    codeforces_store = InMemoryRunStore()

    frontierscience_result = run("frontierscience", store=frontierscience_store)
    codeforces_result = run("codeforces", store=codeforces_store)

    assert frontierscience_result.status is RunStatus.COMPLETED
    assert codeforces_result.status is RunStatus.COMPLETED
