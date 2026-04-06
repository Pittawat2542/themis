from __future__ import annotations

from themis.catalog import get_benchmark, list_benchmark_ids, list_benchmarks


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
    humaneval = entries["humaneval"]

    assert rolebench.base_benchmark_id == "rolebench"
    assert rolebench.declared_variants == [
        "instruction_generalization_eng",
        "role_generalization_eng",
    ]
    assert rolebench.source_kind == "huggingface_raw_files"
    assert livecodebench.dataset_revision == "release_v6"
    assert livecodebench.support_tier == "ready"
    assert humaneval.support_tier == "ready"
    assert humaneval.requires_code_execution is True


def test_get_benchmark_returns_metadata_for_a_single_manifest_entry() -> None:
    benchmark = get_benchmark("livecodebench")

    assert benchmark.benchmark_id == "livecodebench"
    assert benchmark.base_benchmark_id == "livecodebench"
    assert benchmark.source_kind == "huggingface_raw_files"
    assert benchmark.version_notes == "Targets LiveCodeBench release_v6."
