from __future__ import annotations

from themis.catalog import get_benchmark, list_benchmark_ids, list_benchmarks
from themis.catalog.benchmarks import _materialization_benchmark_ids


def test_list_benchmark_ids_exposes_manifest_entries_only() -> None:
    benchmark_ids = list_benchmark_ids()

    assert "mmlu_pro" in benchmark_ids
    assert "livecodebench" in benchmark_ids
    assert "builtin/choice_letter" not in benchmark_ids
    assert benchmark_ids == sorted(benchmark_ids)


def test_list_benchmarks_exposes_structured_benchmark_metadata() -> None:
    entries = {entry.benchmark_id: entry for entry in list_benchmarks()}

    rolebench = entries["rolebench"]
    livecodebench = entries["livecodebench"]
    humaneval_plus = entries["humaneval_plus"]

    assert rolebench.base_benchmark_id == "rolebench"
    assert rolebench.declared_variants == [
        "instruction_generalization_eng",
        "role_generalization_eng",
    ]
    assert rolebench.source_kind == "huggingface_raw_files"
    assert livecodebench.dataset_revision == "release_v6"
    assert livecodebench.support_tier == "ready"
    assert humaneval_plus.support_tier == "ready"
    assert humaneval_plus.requires_code_execution is True


def test_get_benchmark_returns_metadata_for_a_single_manifest_entry() -> None:
    benchmark = get_benchmark("livecodebench")

    assert benchmark.benchmark_id == "livecodebench"
    assert benchmark.base_benchmark_id == "livecodebench"
    assert benchmark.source_kind == "huggingface_raw_files"
    assert benchmark.version_notes == "Targets LiveCodeBench release_v6."


def test_materialization_benchmark_ids_expand_variants_with_examples() -> None:
    benchmark_ids = _materialization_benchmark_ids()

    assert len(benchmark_ids) == 33
    assert benchmark_ids == sorted(benchmark_ids)
    assert set(list_benchmark_ids()).issubset(benchmark_ids)
    assert "rolebench:instruction_generalization_eng" in benchmark_ids
    assert "rolebench:role_generalization_eng" in benchmark_ids
    assert "superchem:en" in benchmark_ids
    assert "superchem:zh" in benchmark_ids
    assert "hle:math,reasoning" in benchmark_ids
    assert "humaneval_plus" in benchmark_ids
    assert "humaneval" not in benchmark_ids
    assert "humaneval_plus:noextreme" not in benchmark_ids
    assert "mmmlu:ZH_CN" in benchmark_ids
    assert "procbench:task07" in benchmark_ids
