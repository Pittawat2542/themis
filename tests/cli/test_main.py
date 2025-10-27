import json
from pathlib import Path

import pytest

from themis.cli import main as cli_main


def create_local_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "math" / "algebra"
    dataset_dir.mkdir(parents=True)
    sample_path = dataset_dir / "sample.json"
    payload = {
        "problem": "What is 2 + 3?",
        "solution": "Add the numbers",
        "answer": "5",
        "subject": "arithmetic",
        "level": 1,
        "unique_id": "cli-1",
    }
    sample_path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def create_mcq_dataset(tmp_path: Path, *, filename: str, answer: str = "A") -> Path:
    dataset_dir = tmp_path / "mcq"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sample_path = dataset_dir / filename
    payload = {
        "question": "Which option is correct?",
        "choices": ["Option A", "Option B", "Option C"],
        "answer": answer,
        "subject": "trivia",
    }
    sample_path.write_text(json.dumps(payload), encoding="utf-8")
    return dataset_dir


def create_competition_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "competition"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sample_path = dataset_dir / "row.json"
    payload = {
        "problem": "Compute 2 + 5",
        "solution": "Add the numbers",
        "answer": "7",
        "subject": "algebra",
    }
    sample_path.write_text(json.dumps(payload), encoding="utf-8")
    return dataset_dir


def test_math500_cli_with_local_dataset(tmp_path, capsys):
    data_root = create_local_dataset(tmp_path)
    storage_dir = tmp_path / "storage"
    args = [
        "math500",
        "--source",
        "local",
        "--data-dir",
        str(data_root),
        "--limit",
        "1",
        "--storage",
        str(storage_dir),
        "--run-id",
        "cli-test",
    ]

    exit_code = cli_main.main(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "exact match" in captured.out.lower()


def test_demo_cli(capsys):
    args = ["demo", "--max-samples", "1"]

    exit_code = cli_main.main(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "evaluated" in captured.out.lower()


def test_supergpqa_cli_with_local_dataset(tmp_path, capsys):
    data_dir = create_mcq_dataset(tmp_path, filename="super.json")
    args = [
        "supergpqa",
        "--source",
        "local",
        "--data-dir",
        str(data_dir),
        "--limit",
        "1",
        "--run-id",
        "cli-super",
    ]

    exit_code = cli_main.main(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "accuracy" in captured.out.lower()


def test_mmlu_pro_cli_with_local_dataset(tmp_path, capsys):
    data_dir = create_mcq_dataset(tmp_path, filename="mmlu.json", answer="B")
    args = [
        "mmlu-pro",
        "--source",
        "local",
        "--data-dir",
        str(data_dir),
        "--limit",
        "1",
        "--run-id",
        "cli-mmlu",
    ]

    exit_code = cli_main.main(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "accuracy" in captured.out.lower()


@pytest.mark.parametrize(
    "command",
    ["aime24", "aime25", "amc23", "olympiadbench", "beyondaime"],
)
def test_competition_cli_with_local_dataset(tmp_path, capsys, command):
    data_dir = create_competition_dataset(tmp_path)
    args = [
        command,
        "--source",
        "local",
        "--data-dir",
        str(data_dir),
        "--limit",
        "1",
        "--run-id",
        f"cli-{command}",
    ]

    exit_code = cli_main.main(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "exact match" in captured.out.lower()
