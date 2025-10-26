import json
from pathlib import Path

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
