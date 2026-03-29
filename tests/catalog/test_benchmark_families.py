from __future__ import annotations

from themis.catalog import load, run
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore


def test_rolebench_variant_uses_variant_specific_rubric() -> None:
    benchmark = load("rolebench:instruction_generalization_eng")
    experiment = benchmark.build_experiment()

    assert benchmark.metric_ids == ["builtin/llm_rubric"]
    assert benchmark.workflow_overrides["rubric"].startswith("Judge whether the response follows the requested role behavior")
    assert benchmark.sample_case_metadata["variant"] == "instruction_generalization_eng"
    assert experiment.evaluation.metrics == ["builtin/llm_rubric"]


def test_procbench_variant_exposes_task_specific_metadata() -> None:
    benchmark = load("procbench:task07")
    experiment = benchmark.build_experiment()

    assert benchmark.metric_ids == ["builtin/llm_rubric"]
    assert benchmark.sample_case_metadata["task_id"] == "task07"
    assert benchmark.sample_case_input["task"].startswith("Complete procbench task07")
    assert experiment.datasets[0].metadata["benchmark_id"] == "procbench:task07"


def test_superchem_and_mmmlu_variants_propagate_language_metadata() -> None:
    superchem = load("superchem:zh")
    mmmlu = load("mmmlu:thai")

    assert superchem.sample_case_metadata["language"] == "zh"
    assert superchem.sample_case_input["language"] == "zh"
    assert mmmlu.sample_case_metadata["language_config"] == "thai"
    assert mmmlu.sample_case_input["language"] == "thai"


def test_hle_and_humaneval_variants_preserve_variant_shapes() -> None:
    hle = load("hle:math,reasoning")
    humaneval = load("humaneval:v0.1.0")
    humaneval_plus = load("humaneval_plus:noextreme")

    assert hle.metric_ids == ["builtin/panel_of_judges"]
    assert hle.sample_case_metadata["domains"] == "math,reasoning"
    assert humaneval.requires_code_execution is True
    assert humaneval.sample_case_metadata["variant"] == "v0.1.0"
    assert humaneval_plus.sample_case_metadata["variant"] == "noextreme"


def test_catalog_load_covers_all_benchmark_entries_from_catalog_md() -> None:
    benchmark_ids = [
        "aime_2025",
        "aime_2026",
        "aethercode",
        "apex_2025",
        "babe",
        "beyond_aime",
        "encyclo_k",
        "frontierscience",
        "gpqa_diamond",
        "healthbench",
        "hle:math,reasoning",
        "hmmt_feb_2025",
        "hmmt_nov_2025",
        "humaneval:mini",
        "humaneval_plus:noextreme",
        "imo_answerbench",
        "livecodebench",
        "lpfqa",
        "mmlu_pro",
        "mmmlu:thai",
        "codeforces",
        "phybench",
        "procbench:task07",
        "rolebench:role_generalization_eng",
        "simpleqa_verified",
        "superchem:en",
        "supergpqa",
    ]

    loaded = [load(benchmark_id) for benchmark_id in benchmark_ids]

    assert [benchmark.benchmark_id for benchmark in loaded] == benchmark_ids


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
