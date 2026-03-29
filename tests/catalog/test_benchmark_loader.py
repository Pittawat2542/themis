from __future__ import annotations

import pytest

from themis.catalog import load


def test_catalog_load_returns_benchmark_definition_for_manifest_entry() -> None:
    benchmark = load("mmlu_pro")

    assert benchmark.benchmark_id == "mmlu_pro"
    assert benchmark.dataset_id == "TIGER-Lab/MMLU-Pro"
    assert benchmark.split == "test"
    assert benchmark.metric_ids == ["builtin/exact_match"]


def test_catalog_load_supports_declared_variants() -> None:
    benchmark = load("rolebench:instruction_generalization_eng")

    assert benchmark.benchmark_id == "rolebench:instruction_generalization_eng"
    assert benchmark.base_benchmark_id == "rolebench"
    assert benchmark.variant == "instruction_generalization_eng"


def test_catalog_load_rejects_invalid_variants() -> None:
    with pytest.raises(ValueError, match="rolebench"):
        load("rolebench:not-a-real-variant")
