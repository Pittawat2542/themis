from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_init_scaffolds_minimal_project(tmp_path: Path) -> None:
    project_root = tmp_path / "demo-project"

    result = subprocess.run(
        [sys.executable, "-m", "themis.cli", "init", "--path", str(project_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (project_root / "experiment.yaml").is_file()
    assert (project_root / "data" / "sample.jsonl").is_file()
    assert (project_root / "run.py").is_file()
    assert 'Path(__file__).with_name("experiment.yaml")' in (project_root / "run.py").read_text()
