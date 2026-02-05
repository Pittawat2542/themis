from __future__ import annotations

import json
from pathlib import Path

import pytest

from themis.cli import main as cli_main


def test_eval_cli_runs_with_jsonl_custom_dataset(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "question": "2+2", "answer": "4"}),
                json.dumps({"id": "2", "question": "1+1", "answer": "2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli_main.eval(
        str(dataset_path),
        model="fake:fake-math-llm",
        prompt="Solve: {question}",
        workers=1,
        storage=str(tmp_path / "storage"),
        resume=False,
    )

    assert exit_code == 0


def test_custom_dataset_loader_rejects_non_object_rows(tmp_path):
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        cli_main._load_custom_dataset_file(dataset_path)


def test_custom_dataset_loader_requires_prompt_and_reference_fields(tmp_path):
    dataset_path = tmp_path / "missing-fields.json"
    dataset_path.write_text(
        json.dumps([{"id": "1", "title": "Only title"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not detect prompt field"):
        cli_main._load_custom_dataset_file(dataset_path)


def test_custom_dataset_loader_requires_reference_field(tmp_path):
    dataset_path = tmp_path / "missing-reference.json"
    dataset_path.write_text(
        json.dumps([{"id": "1", "question": "Only prompt"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not detect reference field"):
        cli_main._load_custom_dataset_file(dataset_path)


def test_custom_dataset_loader_normalizes_reference_alias(tmp_path):
    dataset_path = tmp_path / "expected.jsonl"
    dataset_path.write_text(
        json.dumps({"sample_id": "a", "question": "x", "expected": "y"}) + "\n",
        encoding="utf-8",
    )

    rows, prompt_field, reference_field = cli_main._load_custom_dataset_file(dataset_path)
    assert prompt_field == "question"
    assert reference_field == "expected"
    assert rows[0]["id"] == "a"
    assert rows[0]["reference"] == "y"
