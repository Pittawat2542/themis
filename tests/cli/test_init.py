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
    settings_module = (project_root / "starter_eval" / "settings.py").read_text()
    assert "themis quickcheck scores" in readme
    assert "themis report" in readme
    assert "class CatalogSettings" in settings_module
    assert "THEMIS_CATALOG_PROVIDER" in settings_module
    assert "THEMIS_CATALOG_MODEL" in settings_module


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


def test_init_generates_builtin_benchmark_scaffold(tmp_path: Path) -> None:
    project_root = tmp_path / "starter_eval"

    assert main(["init", str(project_root), "--benchmark", "mmlu_pro"]) == 0

    benchmark_module = (
        project_root / "starter_eval" / "benchmarks" / "default.py"
    ).read_text()
    dataset_module = (
        project_root / "starter_eval" / "datasets" / "builtin.py"
    ).read_text()
    readme = (project_root / "README.md").read_text()

    assert "get_catalog_benchmark" in benchmark_module
    assert "mmlu_pro" in benchmark_module
    assert "get_catalog_benchmark" in dataset_module
    assert "THEMIS_CATALOG_BENCHMARK" in (project_root / ".env.example").read_text()
    assert "themis report" in readme
