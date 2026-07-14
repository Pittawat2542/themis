from __future__ import annotations

import sys
from typing import cast

import pytest

from themis.catalog import load
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.catalog.loaders import (
    BenchmarkSourceRequest,
    MissingOptionalDependencyError,
    load_benchmark_rows,
    load_huggingface_rows,
    load_huggingface_raw_rows,
    load_symbol,
    load_yaml,
)


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


def test_loader_rejects_invalid_symbol_targets() -> None:
    with pytest.raises(ValueError, match="Invalid load target"):
        load_symbol("themis.catalog.loaders")


def test_yaml_loader_requires_mapping_payload(tmp_path) -> None:
    path = tmp_path / "not-a-map.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping config"):
        load_yaml(path)


def test_raw_benchmark_loading_requires_files() -> None:
    with pytest.raises(ValueError, match="No raw files configured"):
        load_huggingface_raw_rows("demo", files=[])


def test_load_benchmark_rows_rejects_unknown_source_kind() -> None:
    request = BenchmarkSourceRequest(
        source_kind="unsupported",
        dataset_id="demo",
        split="test",
    )

    with pytest.raises(ValueError, match="Unknown benchmark source kind"):
        load_benchmark_rows(request)


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


def test_huggingface_dataset_loading_reports_missing_datasets_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        load_huggingface_rows("demo", "test")

    assert "optional datasets dependency" in str(exc_info.value)


def test_huggingface_dataset_loading_streams_rows_as_plain_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeDataset:
        def __iter__(self):
            yield {"question": "one", "answer": "A"}
            yield {"question": "two", "answer": "B"}

    class _FakeDatasetsModule:
        @staticmethod
        def load_dataset(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeDataset()

    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: _FakeDatasetsModule() if name == "datasets" else None,
    )

    rows = load_huggingface_rows(
        "openai/MMMLU",
        "test",
        config_name="ZH_CN",
    )

    assert rows == [
        {"question": "one", "answer": "A"},
        {"question": "two", "answer": "B"},
    ]
    assert captured == {
        "args": ("openai/MMMLU", "ZH_CN"),
        "kwargs": {"split": "test", "revision": None, "streaming": True},
    }


def test_load_benchmark_rows_dispatches_huggingface_dataset_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDataset:
        def __iter__(self):
            yield {"question": "one", "answer": "A"}

    class _FakeDatasetsModule:
        @staticmethod
        def load_dataset(*args, **kwargs):
            del args, kwargs
            return _FakeDataset()

    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: _FakeDatasetsModule() if name == "datasets" else None,
    )

    rows = load_benchmark_rows(
        BenchmarkSourceRequest(
            dataset_id="demo",
            split="test",
            config_name="default",
        )
    )

    assert rows == [{"question": "one", "answer": "A"}]


def test_huggingface_raw_loading_supports_jsonl_rows(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        '{"question": "one", "answer": "A"}\n\n{"question": "two", "answer": "B"}\n',
        encoding="utf-8",
    )

    class _FakeHubModule:
        @staticmethod
        def hf_hub_download(*, repo_id, filename, repo_type, revision=None):
            del repo_id, filename, repo_type, revision
            return str(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: _FakeHubModule() if name == "huggingface_hub" else None,
    )
    try:
        rows = load_huggingface_raw_rows("demo", files=["sample.jsonl"])
    finally:
        monkeypatch.undo()

    assert rows == [
        {"question": "one", "answer": "A"},
        {"question": "two", "answer": "B"},
    ]


def test_huggingface_raw_loading_rejects_non_object_jsonl_rows(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    class _FakeHubModule:
        @staticmethod
        def hf_hub_download(*, repo_id, filename, repo_type, revision=None):
            del repo_id, filename, repo_type, revision
            return str(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: _FakeHubModule() if name == "huggingface_hub" else None,
    )
    try:
        with pytest.raises(ValueError, match="Expected JSON object rows"):
            load_huggingface_raw_rows("demo", files=["sample.jsonl"])
    finally:
        monkeypatch.undo()


def test_huggingface_raw_loading_rejects_unsupported_file_types(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("plain text\n", encoding="utf-8")

    class _FakeHubModule:
        @staticmethod
        def hf_hub_download(*, repo_id, filename, repo_type, revision=None):
            del repo_id, filename, repo_type, revision
            return str(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: _FakeHubModule() if name == "huggingface_hub" else None,
    )
    try:
        with pytest.raises(ValueError, match="Unsupported raw benchmark file type"):
            load_huggingface_raw_rows("demo", files=["sample.txt"])
    finally:
        monkeypatch.undo()


def test_huggingface_raw_loading_supports_parquet_rows(tmp_path) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "question_en": ["What is shown?"],
            "answer_en": [["B"]],
            "question_type": ["multiple_choice"],
        }
    )
    pq.write_table(table, path)

    class _FakeHubModule:
        @staticmethod
        def hf_hub_download(*, repo_id, filename, repo_type, revision=None):
            del repo_id, filename, repo_type, revision
            return str(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "themis.catalog.loaders.importlib.import_module",
        lambda name: _FakeHubModule() if name == "huggingface_hub" else None,
    )
    try:
        rows = load_huggingface_raw_rows("demo", files=["sample.parquet"])
    finally:
        monkeypatch.undo()

    assert rows == [
        {
            "question_en": "What is shown?",
            "answer_en": ["B"],
            "question_type": "multiple_choice",
        }
    ]
