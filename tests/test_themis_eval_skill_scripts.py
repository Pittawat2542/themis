from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / ".agents"
    / "skills"
    / "themis-eval"
    / "scripts"
    / "generate_project_structure.py"
)


def _run_script(tmp_path: Path, *, mode: str) -> Path:
    target = tmp_path / f"{mode}_starter"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--target",
            str(target),
            "--package-name",
            "starter_eval",
            "--mode",
            mode,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    return target


def test_generate_project_structure_creates_default_local_starter_layout(
    tmp_path: Path,
) -> None:
    target = _run_script(tmp_path, mode="default")

    assert (target / "project.toml").exists()
    assert (target / ".env.example").exists()
    assert (target / "README.md").exists()
    assert (target / "data" / "sample.jsonl").exists()
    assert (target / "starter_eval" / "__init__.py").exists()
    assert (target / "starter_eval" / "__main__.py").exists()
    assert (target / "starter_eval" / "app.py").exists()
    assert (target / "starter_eval" / "settings.py").exists()
    assert (target / "starter_eval" / "registry.py").exists()
    assert (target / "starter_eval" / "benchmarks" / "default.py").exists()
    assert (target / "starter_eval" / "datasets" / "local_file.py").exists()


def test_generate_project_structure_creates_builtin_starter_layout(
    tmp_path: Path,
) -> None:
    target = _run_script(tmp_path, mode="builtin")

    assert (target / "project.toml").exists()
    assert (target / ".env.example").exists()
    assert (target / "README.md").exists()
    assert not (target / "data").exists()
    assert (target / "starter_eval" / "__init__.py").exists()
    assert (target / "starter_eval" / "__main__.py").exists()
    assert (target / "starter_eval" / "app.py").exists()
    assert (target / "starter_eval" / "settings.py").exists()
    assert (target / "starter_eval" / "registry.py").exists()
    assert (target / "starter_eval" / "benchmarks" / "default.py").exists()
    assert (target / "starter_eval" / "datasets" / "builtin.py").exists()
