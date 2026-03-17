from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT_EXPORTS = {
    "__version__",
    "BenchmarkResult",
    "BenchmarkSpec",
    "DatasetQuerySpec",
    "ExecutionPolicySpec",
    "InferenceGridSpec",
    "InferenceParamsSpec",
    "ModelSpec",
    "Orchestrator",
    "ParseSpec",
    "PluginRegistry",
    "PostgresBlobStorageSpec",
    "ProjectSpec",
    "PromptMessage",
    "PromptVariantSpec",
    "ScoreSpec",
    "SliceSpec",
    "SqliteBlobStorageSpec",
    "StorageConfig",
    "StorageSpec",
    "generate_config_report",
}


def _load_module(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_release_fixture(
    repo_root: Path,
    *,
    version: str = "2.0.0",
    changelog_date: str = "2026-03-10",
    citation_date: str = "2026-03-10",
    classifier: str = "Development Status :: 5 - Production/Stable",
) -> None:
    (repo_root / "docs" / "changelog").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text(
        f'[project]\nversion = "{version}"\nclassifiers = [\n    "{classifier}",\n]\n'
    )
    (repo_root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{version}] - {changelog_date}\n"
    )
    (repo_root / "docs" / "changelog" / "index.md").write_text(
        "# Changelog\n\n"
        "The canonical release history lives in the repository root:\n\n"
        "- [Root changelog on GitHub](https://github.com/Pittawat2542/themis/blob/main/CHANGELOG.md)\n"
    )
    (repo_root / "CITATION.cff").write_text(
        f"date-released: {citation_date}\nversion: {version}\n"
    )


def test_validate_release_uses_docs_changelog_index_link(tmp_path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    repo_root = tmp_path / "repo"
    _write_release_fixture(repo_root)

    failures = module.collect_validation_failures(repo_root, tag="v2.0.0")

    assert failures == []


def test_validate_release_reports_missing_docs_index_cleanly(tmp_path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nversion = "2.0.0"\nclassifiers = ["Development Status :: 5 - Production/Stable"]\n'
    )
    (repo_root / "CHANGELOG.md").write_text("# Changelog\n\n## [2.0.0] - 2026-03-10\n")
    (repo_root / "CITATION.cff").write_text(
        "date-released: 2026-03-10\nversion: 2.0.0\n"
    )

    failures = module.collect_validation_failures(repo_root, tag="v2.0.0")

    assert failures == ["docs/changelog/index.md is missing"]


def test_validate_release_rejects_mismatched_citation_release_date(tmp_path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    repo_root = tmp_path / "repo"
    _write_release_fixture(repo_root, citation_date="2026-03-11")

    failures = module.collect_validation_failures(repo_root, tag="v2.0.0")

    assert failures == [
        "CITATION.cff date-released '2026-03-11' does not match CHANGELOG.md date '2026-03-10' for version [2.0.0]"
    ]


def test_validate_release_rejects_beta_classifier_for_stable_release(tmp_path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    repo_root = tmp_path / "repo"
    _write_release_fixture(repo_root, classifier="Development Status :: 4 - Beta")

    failures = module.collect_validation_failures(repo_root, tag="v2.0.0")

    assert failures == [
        "pyproject.toml must declare 'Development Status :: 5 - Production/Stable' for stable release 2.0.0"
    ]


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


def test_check_built_package_uses_v3_root_exports() -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    assert module.EXPECTED_ROOT_EXPORTS == EXPECTED_ROOT_EXPORTS


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
    assert "PromptRole.SYSTEM:Be concise and explicit." in captured.out
    assert "question=Explain what you are doing." in captured.out
