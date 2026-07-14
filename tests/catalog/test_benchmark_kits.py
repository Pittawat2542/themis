from __future__ import annotations

from themis.catalog import (
    BenchmarkExperimentDefaults,
    BenchmarkKit,
    build_benchmark_experiment,
    get_benchmark_kit,
    list_benchmark_kits,
)
from themis import Experiment, RunOptions


def test_benchmark_kit_discovery_returns_themis_owned_kits() -> None:
    kit_ids = list_benchmark_kits()

    assert "mmlu_pro" in kit_ids
    assert "frontierscience" in kit_ids
    assert "humaneval_plus" in kit_ids

    kit = get_benchmark_kit("mmlu_pro")

    assert isinstance(kit, BenchmarkKit)
    assert kit.kit_id == "benchmark/mmlu_pro"
    assert kit.benchmark_id == "mmlu_pro"
    assert isinstance(kit.defaults, BenchmarkExperimentDefaults)


def test_benchmark_kit_builds_complete_experiment_with_overrides(
    catalog_fixture_loader: None,
) -> None:
    experiment = build_benchmark_experiment(
        "mmlu_pro",
        overrides=BenchmarkExperimentDefaults(
            generator="builtin/demo_generator",
            samples=2,
            reducer="builtin/majority_vote",
        ),
    )

    assert isinstance(experiment, Experiment)

    snapshot = experiment.compile(options=RunOptions(max_concurrency=3))

    assert snapshot.identity.candidate_policy["num_samples"] == 2
    assert snapshot.component_refs.generator.component_id == "builtin/demo_generator"
    assert snapshot.component_refs.metrics
    assert snapshot.component_refs.parsers
    assert snapshot.dataset_sources[0].target == "catalog"


def test_runtime_only_kit_build_settings_do_not_change_run_identity(
    catalog_fixture_loader: None,
) -> None:
    first = build_benchmark_experiment("mmlu_pro")
    second = build_benchmark_experiment("mmlu_pro")

    assert (
        first.compile(options=RunOptions(max_concurrency=1)).run_id
        == second.compile(options=RunOptions(max_concurrency=9)).run_id
    )
