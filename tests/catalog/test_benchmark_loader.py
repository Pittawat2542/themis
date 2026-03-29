from __future__ import annotations

import pytest

from themis.catalog import load


def test_catalog_load_returns_benchmark_definition_for_manifest_entry() -> None:
    benchmark = load("mmlu_pro")

    assert benchmark.benchmark_id == "mmlu_pro"
    assert benchmark.dataset_id == "TIGER-Lab/MMLU-Pro"
    assert benchmark.split == "test"
    assert benchmark.metric_ids == ["builtin/exact_match"]
    assert benchmark.requires_code_execution is False


def test_catalog_load_supports_declared_variants() -> None:
    benchmark = load("rolebench:instruction_generalization_eng")

    assert benchmark.benchmark_id == "rolebench:instruction_generalization_eng"
    assert benchmark.base_benchmark_id == "rolebench"
    assert benchmark.variant == "instruction_generalization_eng"


def test_catalog_load_rejects_invalid_variants() -> None:
    with pytest.raises(ValueError, match="rolebench"):
        load("rolebench:not-a-real-variant")


def test_catalog_load_marks_code_benchmarks_and_supported_backends() -> None:
    benchmark = load("codeforces")

    assert benchmark.benchmark_id == "codeforces"
    assert benchmark.dataset_revision == "verifiable-prompts"
    assert benchmark.requires_code_execution is True
    assert benchmark.supported_execution_backends == ["piston", "sandbox_fusion"]


def test_catalog_load_preserves_dataset_revisions_from_catalog_notes() -> None:
    aethercode = load("aethercode")
    livecodebench = load("livecodebench")

    assert aethercode.dataset_revision == "v1_2024"
    assert livecodebench.dataset_revision == "release_v6"


def test_catalog_manifest_covers_representative_benchmark_families() -> None:
    benchmark_ids = [
        "aime_2025",
        "gpqa_diamond",
        "mmlu_pro",
        "rolebench:role_generalization_eng",
        "superchem:en",
        "humaneval:mini",
        "procbench:task07",
        "simpleqa_verified",
    ]

    loaded = [load(benchmark_id) for benchmark_id in benchmark_ids]

    assert [benchmark.benchmark_id for benchmark in loaded] == benchmark_ids
