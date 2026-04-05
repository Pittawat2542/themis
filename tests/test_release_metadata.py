from __future__ import annotations

from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, object]:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert isinstance(project, dict)
    return project


def _lock_packages() -> list[dict[str, object]]:
    payload = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = payload["package"]
    assert isinstance(packages, list)
    return packages


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_pyproject_has_release_ready_public_metadata() -> None:
    project = _project_metadata()

    assert project["name"] == "themis-eval"
    assert project["version"] == "4.0.0"
    assert "v4" not in str(project["description"]).lower()
    assert project["requires-python"] == ">=3.12"
    assert project["license"] == "MIT"

    urls = project["urls"]
    assert isinstance(urls, dict)
    for key in ("Repository", "Homepage", "Documentation", "Changelog", "Issues"):
        assert key in urls

    keywords = project["keywords"]
    assert isinstance(keywords, list)
    for keyword in ("llm", "evaluation", "benchmark", "research"):
        assert keyword in keywords

    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)
    for classifier in (
        "Programming Language :: Python :: 3 :: Only",
        "Typing :: Typed",
    ):
        assert classifier in classifiers


def test_runtime_dependencies_exclude_docs_tooling_and_library_conflict_pins() -> None:
    project = _project_metadata()

    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    dependency_text = "\n".join(str(dependency) for dependency in dependencies)
    assert "mkdocs" not in dependency_text
    assert "mkdocstrings" not in dependency_text

    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    assert "docs" in optional
    docs_dependencies = "\n".join(str(dependency) for dependency in optional["docs"])
    assert "mkdocs" in docs_dependencies
    assert "mkdocstrings" in docs_dependencies

    openai_dependencies = optional["openai"]
    assert isinstance(openai_dependencies, list)
    assert all("<" not in str(dependency) for dependency in openai_dependencies)


def test_readme_uses_public_install_instructions_and_no_repo_local_links() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "# Themis" in readme
    assert "Themis v4" not in readme
    assert "uv add themis-eval" in readme
    assert "/Users/" not in readme
    assert 'uv pip install -e ".[dev]"' not in readme


def test_lockfile_excludes_vulnerable_anthropic_release() -> None:
    anthropic_versions = {
        str(package["version"])
        for package in _lock_packages()
        if package.get("name") == "anthropic"
    }

    assert anthropic_versions
    assert all(
        _parse_version(version) < (0, 86, 0) or _parse_version(version) >= (0, 87, 0)
        for version in anthropic_versions
    )
