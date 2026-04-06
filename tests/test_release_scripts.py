from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

from tests.release import (
    CURRENT_DIST_BASENAME,
    CURRENT_DIST_INFO,
    CURRENT_SDIST,
    CURRENT_TAG,
    CURRENT_VERSION,
    CURRENT_WHEEL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_ci_script(tmp_path: Path, script_name: str) -> Path:
    repo_root = tmp_path / "repo"
    script_dir = repo_root / "scripts" / "ci"
    script_dir.mkdir(parents=True)
    source = PROJECT_ROOT / "scripts" / "ci" / script_name
    target = script_dir / script_name
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_validate_release_extracts_version_headings(tmp_path: Path) -> None:
    module = _load_module("scripts/ci/validate_release.py", "validate_release")

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        f"## [{CURRENT_VERSION}] - 2026-04-06\n"
        "Stable release.\n\n"
        "## [Unreleased]\n"
        "Draft.\n\n"
        "## [3.9.0] - 2026-04-01\n"
        "Older release.\n",
        encoding="utf-8",
    )

    assert module._extract_versions(changelog) == {CURRENT_VERSION, "3.9.0"}


def test_validate_release_main_accepts_matching_tag(tmp_path: Path) -> None:
    script = _copy_ci_script(tmp_path, "validate_release.py")
    repo_root = script.parents[2]
    (repo_root / "pyproject.toml").write_text(
        f'[project]\nversion = "{CURRENT_VERSION}"\n',
        encoding="utf-8",
    )
    (repo_root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{CURRENT_VERSION}] - 2026-04-06\nRelease notes.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script), "--tag", CURRENT_TAG],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"release metadata validated for {CURRENT_TAG}" in result.stdout


def test_validate_release_main_rejects_missing_changelog_section(
    tmp_path: Path,
) -> None:
    script = _copy_ci_script(tmp_path, "validate_release.py")
    repo_root = script.parents[2]
    (repo_root / "pyproject.toml").write_text(
        f'[project]\nversion = "{CURRENT_VERSION}"\n',
        encoding="utf-8",
    )
    (repo_root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [3.9.0] - 2026-04-01\nOlder release.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script), "--tag", CURRENT_TAG],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        f"does not contain a release section for version {CURRENT_VERSION}"
        in result.stderr
    )


def test_extract_release_notes_writes_requested_section(tmp_path: Path) -> None:
    script = _copy_ci_script(tmp_path, "extract_release_notes.py")
    changelog = tmp_path / "CHANGELOG.md"
    output = tmp_path / "release_notes.md"
    changelog.write_text(
        "# Changelog\n\n"
        f"## [{CURRENT_VERSION}] - 2026-04-06\n"
        "Line one.\n"
        "Line two.\n\n"
        "## [3.9.0] - 2026-04-01\n"
        "Older release.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tag",
            CURRENT_TAG,
            "--changelog",
            str(changelog),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == (
        f"## [{CURRENT_VERSION}] - 2026-04-06\nLine one.\nLine two.\n"
    )


def test_check_built_package_rejects_excluded_files() -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    path = Path(CURRENT_WHEEL)
    try:
        module._assert_archive_contents(path, ["themis/__init__.py", "docs/index.md"])
    except SystemExit as exc:
        assert "unexpectedly contains excluded path: docs/index.md" in str(exc)
    else:
        raise AssertionError("expected excluded archive path failure")


def test_check_built_package_inspect_wheel_requires_cli_entry_point(
    tmp_path: Path,
) -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    wheel = tmp_path / CURRENT_WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("themis/__init__.py", "__all__ = []\n")
        archive.writestr(
            f"{CURRENT_DIST_INFO}/METADATA",
            f"Metadata-Version: 2.4\nName: themis-eval\nVersion: {CURRENT_VERSION}\n",
        )
        archive.writestr(
            f"{CURRENT_DIST_INFO}/entry_points.txt",
            "[console_scripts]\nother = themis.cli:main\n",
        )

    try:
        module._inspect_wheel(wheel)
    except SystemExit as exc:
        assert "missing the themis CLI entry point" in str(exc)
    else:
        raise AssertionError("expected missing entry point failure")


def test_check_built_package_inspect_sdist_accepts_package_layout(
    tmp_path: Path,
) -> None:
    module = _load_module("scripts/ci/check_built_package.py", "check_built_package")

    sdist = tmp_path / CURRENT_SDIST
    with tarfile.open(sdist, "w:gz") as archive:
        package_dir = tmp_path / CURRENT_DIST_BASENAME / "themis"
        package_dir.mkdir(parents=True)
        init_file = package_dir / "__init__.py"
        init_file.write_text("__all__ = []\n", encoding="utf-8")
        archive.add(init_file, arcname=f"{CURRENT_DIST_BASENAME}/themis/__init__.py")

    module._inspect_sdist(sdist)


def test_run_examples_executes_scripts_in_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module("scripts/ci/run_examples.py", "run_examples")

    script = tmp_path / "example.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    root_dir = tmp_path / "repo"
    root_dir.mkdir()

    calls: list[tuple[list[str], Path]] = []

    def _fake_run(cmd, *, cwd, capture_output, text, check=False):
        del check
        calls.append((list(cmd), cwd))
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    result = module._run_example(script, root_dir=root_dir)

    assert result.returncode == 0
    assert calls == [([sys.executable, str(script)], root_dir)]


def test_run_examples_resets_cache_dirs(tmp_path: Path, monkeypatch) -> None:
    module = _load_module("scripts/ci/run_examples.py", "run_examples")

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cache" / "data").mkdir(parents=True)
    (tmp_path / ".themis_cache" / "data").mkdir(parents=True)

    module._reset_example_cache_dirs()

    assert not (tmp_path / ".cache").exists()
    assert not (tmp_path / ".themis_cache").exists()
