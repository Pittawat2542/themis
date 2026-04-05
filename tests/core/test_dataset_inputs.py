from __future__ import annotations

import sys
from pathlib import Path

import pytest

from themis.core.dataset_inputs import (
    MissingOptionalDependencyError,
    dataset_from_huggingface,
    dataset_from_inline,
    dataset_from_jsonl,
)


def test_dataset_from_inline_builds_single_case_dataset() -> None:
    dataset = dataset_from_inline(
        input_value={"question": "2+2"},
        expected_output={"answer": "4"},
    )

    assert dataset.dataset_id == "inline"
    assert dataset.cases[0].input == {"question": "2+2"}
    assert dataset.cases[0].expected_output == {"answer": "4"}


def test_dataset_from_jsonl_reads_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}',
                '{"case_id":"case-2","input":{"question":"3+3"},"expected_output":{"answer":"6"}}',
            ]
        )
    )

    dataset = dataset_from_jsonl(path)

    assert dataset.dataset_id == "cases"
    assert [case.case_id for case in dataset.cases] == ["case-1", "case-2"]


def test_dataset_from_huggingface_raises_clear_error_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(
        "themis.core.dataset_inputs.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )

    with pytest.raises(
        MissingOptionalDependencyError, match='uv add "themis-eval\\[datasets\\]"'
    ):
        dataset_from_huggingface(
            dataset_name="demo",
            split="train",
            input_field="prompt",
            expected_output_field="answer",
        )


def test_dataset_from_huggingface_loads_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDatasetsModule:
        @staticmethod
        def load_dataset(dataset_name: str, *, split: str):
            assert dataset_name == "demo"
            assert split == "train"
            return [
                {
                    "id": "row-1",
                    "prompt": {"question": "2+2"},
                    "answer": {"answer": "4"},
                },
                {
                    "id": "row-2",
                    "prompt": {"question": "3+3"},
                    "answer": {"answer": "6"},
                },
            ]

    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetsModule())

    dataset = dataset_from_huggingface(
        dataset_name="demo",
        split="train",
        input_field="prompt",
        expected_output_field="answer",
        case_id_field="id",
    )

    assert dataset.dataset_id == "demo"
    assert [case.case_id for case in dataset.cases] == ["row-1", "row-2"]
