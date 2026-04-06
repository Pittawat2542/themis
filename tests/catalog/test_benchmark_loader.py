from __future__ import annotations

import sys
from typing import cast

import pytest

from themis.catalog import load
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.catalog.loaders import (
    MissingOptionalDependencyError,
    load_huggingface_raw_rows,
)
from tests.catalog_ids import catalog_benchmark_ids


def test_catalog_load_returns_benchmark_definition_for_manifest_entry() -> None:
    benchmark = cast(BenchmarkDefinition, load("mmlu_pro"))

    assert benchmark.benchmark_id == "mmlu_pro"
    assert benchmark.dataset_id == "TIGER-Lab/MMLU-Pro"
    assert benchmark.split == "test"
    assert benchmark.metric_ids == ["builtin/choice_accuracy"]
    assert benchmark.parser_ids == ["builtin/choice_letter"]
    assert benchmark.requires_code_execution is False


def test_catalog_load_supports_declared_variants() -> None:
    benchmark = cast(
        BenchmarkDefinition, load("rolebench:instruction_generalization_eng")
    )

    assert benchmark.benchmark_id == "rolebench:instruction_generalization_eng"
    assert benchmark.base_benchmark_id == "rolebench"
    assert benchmark.variant == "instruction_generalization_eng"


def test_catalog_load_rejects_invalid_variants() -> None:
    with pytest.raises(ValueError, match="rolebench"):
        load("rolebench:not-a-real-variant")


def test_catalog_load_marks_code_benchmarks_and_supported_backends() -> None:
    benchmark = cast(BenchmarkDefinition, load("codeforces"))

    assert benchmark.benchmark_id == "codeforces"
    assert benchmark.dataset_revision == "verifiable-prompts"
    assert benchmark.requires_code_execution is True
    assert benchmark.supported_execution_backends == ["piston", "sandbox_fusion"]


def test_catalog_load_preserves_dataset_revisions_from_catalog_notes() -> None:
    aethercode = cast(BenchmarkDefinition, load("aethercode"))
    livecodebench = cast(BenchmarkDefinition, load("livecodebench"))

    assert aethercode.dataset_revision == "v1_2024"
    assert livecodebench.dataset_revision == "release_v6"


def test_catalog_manifest_covers_representative_benchmark_families() -> None:
    loaded = [
        cast(BenchmarkDefinition, load(benchmark_id))
        for benchmark_id in catalog_benchmark_ids()
    ]

    assert [benchmark.benchmark_id for benchmark in loaded] == catalog_benchmark_ids()


def test_raw_benchmark_loading_reports_missing_huggingface_hub_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        load_huggingface_raw_rows("demo", files=["train.jsonl"])

    message = str(exc_info.value)
    assert "huggingface_hub" in message
    assert "pip install huggingface-hub" in message
    assert 'uv add "themis-eval[datasets]"' in message
