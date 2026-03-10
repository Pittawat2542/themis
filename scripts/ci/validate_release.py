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
CANONICAL_CHANGELOG_URL = (
    "https://github.com/Pittawat2542/themis/blob/main/CHANGELOG.md"
)


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


def read_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


def extract_versions_from_changelog(path: Path) -> set[str]:
    versions: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CHANGELOG_HEADER_RE.match(line.strip())
        if match:
            versions.add(match.group("version"))
    return versions


def extract_citation_version(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CITATION_VERSION_RE.match(line.strip())
        if match:
            return match.group("version").strip("\"'")
    return None


def docs_changelog_references_root(path: Path) -> bool:
    return CANONICAL_CHANGELOG_URL in path.read_text(encoding="utf-8")


def collect_validation_failures(repo_root: Path, *, tag: str = "") -> list[str]:
    pyproject_path = repo_root / "pyproject.toml"
    changelog_path = repo_root / "CHANGELOG.md"
    docs_changelog_path = repo_root / "docs" / "changelog" / "index.md"
    citation_path = repo_root / "CITATION.cff"

    failures: list[str] = []
    package_version: str | None = None
    changelog_versions: set[str] = set()

    if not pyproject_path.exists():
        failures.append("pyproject.toml is missing")
    else:
        package_version = read_project_version(pyproject_path)
        if not SEMVER_RE.match(package_version):
            failures.append(
                f"pyproject.toml version '{package_version}' is not a valid semantic version"
            )

    if not changelog_path.exists():
        failures.append("CHANGELOG.md is missing")
    else:
        changelog_versions = extract_versions_from_changelog(changelog_path)

    if package_version is not None and package_version not in changelog_versions:
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
        if citation_version is None:
            failures.append("CITATION.cff is missing a version field")
        elif package_version is not None and citation_version != package_version:
            failures.append(
                f"CITATION.cff version '{citation_version}' does not match pyproject version '{package_version}'"
            )

    if tag:
        tag_version = normalize_tag(tag)
        if package_version is not None and tag_version != package_version:
            failures.append(
                f"Release tag '{tag}' resolves to '{tag_version}', but pyproject version is '{package_version}'"
            )
        if tag_version not in changelog_versions:
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
