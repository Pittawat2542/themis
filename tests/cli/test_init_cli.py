from __future__ import annotations

from pathlib import Path

from tests.cli.helpers import run_cli


def test_init_scaffolds_minimal_project(tmp_path: Path) -> None:
    project_root = tmp_path / "demo-project"

    result = run_cli("init", "--path", str(project_root))

    assert result.returncode == 0, result.stderr
    assert (project_root / "experiment.yaml").is_file()
    assert (project_root / "experiment.py").is_file()
    assert (project_root / "data" / "sample.jsonl").is_file()
    assert (project_root / "run.py").is_file()
    assert (
        'from experiment import experiment'
        in (project_root / "run.py").read_text()
    )
