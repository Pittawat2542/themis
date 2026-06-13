from __future__ import annotations

from themis.catalog import (
    build_benchmark_experiment,
    get_benchmark_kit,
    list_benchmark_kits,
)
from themis.catalog.benchmarks import BenchmarkExperimentDefaults, BenchmarkKit
from themis.core.config import RuntimeConfig, StorageConfig
from themis.core.experiment import Experiment


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


def test_benchmark_kit_builds_complete_experiment_with_overrides() -> None:
    experiment = build_benchmark_experiment(
        "mmlu_pro",
        storage=StorageConfig(target="memory"),
        runtime=RuntimeConfig(max_concurrent_tasks=3),
        overrides=BenchmarkExperimentDefaults(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
    )

    assert isinstance(experiment, Experiment)
    assert experiment.runtime.max_concurrent_tasks == 3

    snapshot = experiment.compile()

    assert snapshot.identity.candidate_policy["num_samples"] == 2
    assert snapshot.component_refs.generator.component_id == "builtin/demo_generator"
    assert snapshot.component_refs.metrics
    assert snapshot.component_refs.parsers
    assert snapshot.dataset_sources[0].target == "catalog"


def test_runtime_only_kit_build_settings_do_not_change_run_identity() -> None:
    first = build_benchmark_experiment(
        "mmlu_pro",
        runtime=RuntimeConfig(max_concurrent_tasks=1),
    )
    second = build_benchmark_experiment(
        "mmlu_pro",
        runtime=RuntimeConfig(max_concurrent_tasks=9),
    )

    assert first.compile().run_id == second.compile().run_id
