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
DOCS_VERSION_RE = re.compile(r"^- Version:\s*`(?P<version>[^`]+)`\s*$")
CITATION_VERSION_RE = re.compile(r"^version:\s*(?P<version>[^\s#]+)\s*$")


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


def extract_docs_current_version(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = DOCS_VERSION_RE.match(line.strip())
        if match:
            return match.group("version")
    return None


def extract_citation_version(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CITATION_VERSION_RE.match(line.strip())
        if match:
            return match.group("version").strip("\"'")
    return None


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    pyproject_path = repo_root / "pyproject.toml"
    changelog_path = repo_root / "CHANGELOG.md"
    docs_changelog_path = repo_root / "docs" / "CHANGELOG.md"
    citation_path = repo_root / "CITATION.cff"

    failures: list[str] = []

    package_version = read_project_version(pyproject_path)
    if not SEMVER_RE.match(package_version):
        failures.append(
            f"pyproject.toml version '{package_version}' is not a valid semantic version"
        )

    changelog_versions = extract_versions_from_changelog(changelog_path)
    if package_version not in changelog_versions:
        failures.append(
            f"CHANGELOG.md is missing release heading for version [{package_version}]"
        )

    docs_version = extract_docs_current_version(docs_changelog_path)
    if docs_version is None:
        failures.append("docs/CHANGELOG.md is missing a 'Version: `x.y.z`' entry")
    elif docs_version != package_version:
        failures.append(
            "docs/CHANGELOG.md current version "
            f"'{docs_version}' does not match pyproject version '{package_version}'"
        )

    citation_version = extract_citation_version(citation_path)
    if citation_version is None:
        failures.append("CITATION.cff is missing a version field")
    elif citation_version != package_version:
        failures.append(
            f"CITATION.cff version '{citation_version}' does not match pyproject version '{package_version}'"
        )

    if args.tag:
        tag_version = normalize_tag(args.tag)
        if tag_version != package_version:
            failures.append(
                f"Release tag '{args.tag}' resolves to '{tag_version}', but pyproject version is '{package_version}'"
            )
        if tag_version not in changelog_versions:
            failures.append(
                f"CHANGELOG.md is missing release heading for tag version [{tag_version}]"
            )

    if failures:
        print("Release metadata validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Release metadata is valid for version {package_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
