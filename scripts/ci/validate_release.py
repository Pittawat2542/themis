#!/usr/bin/env python3
"""Validate release metadata consistency across project files."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:\.post\d+)?(?:[-+][0-9A-Za-z.-]+)?$")
CHANGELOG_HEADER_RE = re.compile(
    r"^## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})$"
)
CITATION_VERSION_RE = re.compile(r"^version:\s*(?P<version>[^\s#]+)\s*$")
CITATION_DATE_RE = re.compile(r"^date-released:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$")
CANONICAL_CHANGELOG_URL = (
    "https://github.com/Pittawat2542/themis/blob/main/CHANGELOG.md"
)
STABLE_RELEASE_CLASSIFIER = "Development Status :: 5 - Production/Stable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="",
        help="Release tag to validate (for example: v1.2.3 or 1.2.3)",
    )
    return parser.parse_args()


def normalize_tag(tag: str) -> str:
    normalized = tag.strip()
    if normalized.startswith("refs/tags/"):
        normalized = normalized.removeprefix("refs/tags/")
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized


def read_project_metadata(pyproject_path: Path) -> dict[str, object]:
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    project_data = data.get("project")
    if not isinstance(project_data, dict):
        raise KeyError("project")
    return project_data


def read_project_version(pyproject_path: Path) -> str:
    project_data = read_project_metadata(pyproject_path)
    return str(project_data["version"])


def read_project_classifiers(pyproject_path: Path) -> list[str]:
    project_data = read_project_metadata(pyproject_path)
    classifiers = project_data.get("classifiers", [])
    if not isinstance(classifiers, list):
        return []
    return [str(classifier) for classifier in classifiers]


def extract_release_dates_from_changelog(path: Path) -> dict[str, str]:
    releases: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CHANGELOG_HEADER_RE.match(line.strip())
        if match:
            releases[match.group("version")] = match.group("date")
    return releases


def extract_citation_version(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CITATION_VERSION_RE.match(line.strip())
        if match:
            return match.group("version").strip("\"'")
    return None


def extract_citation_release_date(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CITATION_DATE_RE.match(line.strip())
        if match:
            return match.group("date")
    return None


def is_stable_release(version: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+\.\d+", version))


def docs_changelog_references_root(path: Path) -> bool:
    return CANONICAL_CHANGELOG_URL in path.read_text(encoding="utf-8")


def collect_validation_failures(repo_root: Path, *, tag: str = "") -> list[str]:
    pyproject_path = repo_root / "pyproject.toml"
    changelog_path = repo_root / "CHANGELOG.md"
    docs_changelog_path = repo_root / "docs" / "changelog" / "index.md"
    citation_path = repo_root / "CITATION.cff"

    failures: list[str] = []
    package_version: str | None = None
    project_classifiers: list[str] = []
    changelog_releases: dict[str, str] = {}

    if not pyproject_path.exists():
        failures.append("pyproject.toml is missing")
    else:
        package_version = read_project_version(pyproject_path)
        project_classifiers = read_project_classifiers(pyproject_path)
        if not SEMVER_RE.match(package_version):
            failures.append(
                f"pyproject.toml version '{package_version}' is not a valid semantic version"
            )
        elif is_stable_release(package_version) and (
            STABLE_RELEASE_CLASSIFIER not in project_classifiers
        ):
            failures.append(
                "pyproject.toml must declare "
                f"'{STABLE_RELEASE_CLASSIFIER}' for stable release {package_version}"
            )

    if not changelog_path.exists():
        failures.append("CHANGELOG.md is missing")
    else:
        changelog_releases = extract_release_dates_from_changelog(changelog_path)

    if package_version is not None and package_version not in changelog_releases:
        failures.append(
            f"CHANGELOG.md is missing release heading for version [{package_version}]"
        )

    if not docs_changelog_path.exists():
        failures.append("docs/changelog/index.md is missing")
    elif not docs_changelog_references_root(docs_changelog_path):
        failures.append(
            "docs/changelog/index.md must reference the canonical root changelog"
        )

    if not citation_path.exists():
        failures.append("CITATION.cff is missing")
    else:
        citation_version = extract_citation_version(citation_path)
        citation_release_date = extract_citation_release_date(citation_path)
        if citation_version is None:
            failures.append("CITATION.cff is missing a version field")
        elif package_version is not None and citation_version != package_version:
            failures.append(
                f"CITATION.cff version '{citation_version}' does not match pyproject version '{package_version}'"
            )
        elif package_version is not None:
            changelog_date = changelog_releases.get(package_version)
            if citation_release_date is None:
                failures.append("CITATION.cff is missing a date-released field")
            elif changelog_date is not None and citation_release_date != changelog_date:
                failures.append(
                    "CITATION.cff date-released "
                    f"'{citation_release_date}' does not match CHANGELOG.md date "
                    f"'{changelog_date}' for version [{package_version}]"
                )

    if tag:
        tag_version = normalize_tag(tag)
        if package_version is not None and tag_version != package_version:
            failures.append(
                f"Release tag '{tag}' resolves to '{tag_version}', but pyproject version is '{package_version}'"
            )
        if tag_version not in changelog_releases:
            failures.append(
                f"CHANGELOG.md is missing release heading for tag version [{tag_version}]"
            )

    return failures


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    failures = collect_validation_failures(repo_root, tag=args.tag)

    if failures:
        print("Release metadata validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    package_version = read_project_version(repo_root / "pyproject.toml")
    print(f"Release metadata is valid for version {package_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
