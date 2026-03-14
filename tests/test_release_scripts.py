from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_release_uses_docs_changelog_index_link(tmp_path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    repo_root = tmp_path / "repo"
    (repo_root / "docs" / "changelog").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[project]\nversion = "2.0.0"\n')
    (repo_root / "CHANGELOG.md").write_text("# Changelog\n\n## [2.0.0] - 2026-03-10\n")
    (repo_root / "docs" / "changelog" / "index.md").write_text(
        "# Changelog\n\n"
        "The canonical release history lives in the repository root:\n\n"
        "- [Root changelog on GitHub](https://github.com/Pittawat2542/themis/blob/main/CHANGELOG.md)\n"
    )
    (repo_root / "CITATION.cff").write_text("version: 2.0.0\n")

    failures = module.collect_validation_failures(repo_root, tag="v2.0.0")

    assert failures == []


def test_validate_release_reports_missing_docs_index_cleanly(tmp_path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text('[project]\nversion = "2.0.0"\n')
    (repo_root / "CHANGELOG.md").write_text("# Changelog\n\n## [2.0.0] - 2026-03-10\n")
    (repo_root / "CITATION.cff").write_text("version: 2.0.0\n")

    failures = module.collect_validation_failures(repo_root, tag="v2.0.0")

    assert failures == ["docs/changelog/index.md is missing"]


def test_check_built_package_sanitizes_verification_environment() -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    env = module._verification_env({"PYTHONPATH": "repo", "HOME": "/tmp/home"})

    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["HOME"] == "/tmp/home"


def test_check_built_package_import_check_targets_site_packages() -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    snippet = module._import_check_snippet(
        Path("/tmp/venv/lib/python3.12/site-packages")
    )

    assert "themis.__file__" in snippet
    assert "site-packages" in snippet
    assert "expected = " in snippet


def test_check_built_package_selects_wheel_matching_project_version(tmp_path) -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "themis_eval-1.4.0-py3-none-any.whl").write_text("")
    expected_wheel = dist_dir / "themis_eval-2.0.0-py3-none-any.whl"
    expected_wheel.write_text("")

    wheel = module._select_built_wheel(dist_dir, project_version="2.0.0")

    assert wheel == expected_wheel


def test_hooks_and_timeline_example_runs(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        str(PROJECT_ROOT / "examples" / "06_hooks_and_timeline.py"),
        run_name="__main__",
    )

    captured = capsys.readouterr()
    assert "Telemetry events:" in captured.out
    assert "Candidate output:" in captured.out
