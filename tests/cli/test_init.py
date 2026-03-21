from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from themis.cli.main import main


def test_init_generates_lean_project_scaffold(tmp_path: Path) -> None:
    project_root = tmp_path / "starter_eval"

    assert main(["init", str(project_root)]) == 0

    assert (project_root / "project.toml").exists()
    assert (project_root / ".env.example").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "data" / "sample.jsonl").exists()
    assert (project_root / "starter_eval" / "__main__.py").exists()
    assert (project_root / "starter_eval" / "app.py").exists()
    assert (project_root / "starter_eval" / "registry.py").exists()
    assert (project_root / "starter_eval" / "settings.py").exists()
    assert (project_root / "starter_eval" / "benchmarks" / "__init__.py").exists()
    assert (project_root / "starter_eval" / "datasets" / "__init__.py").exists()

    readme = (project_root / "README.md").read_text()
    assert "themis quickcheck scores" in readme
    assert "themis report" in readme


def test_init_generated_project_runs_preview_mode(tmp_path: Path) -> None:
    project_root = tmp_path / "starter_eval"

    assert main(["init", str(project_root), "--provider", "demo"]) == 0

    completed = subprocess.run(
        [sys.executable, "-m", "starter_eval", "--preview"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Question: What is 2 + 2?" in completed.stdout
