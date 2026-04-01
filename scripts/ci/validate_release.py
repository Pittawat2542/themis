from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib


TAG_PATTERN = re.compile(r"^v(?P<version>[0-9]+(?:\.[0-9]+){2}(?:[A-Za-z0-9.-]+)?)$")


def _extract_versions(changelog_path: Path) -> set[str]:
    versions: set[str] = set()
    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("## "):
            continue
        heading = line[3:].strip()
        version = heading.split(" - ", 1)[0].strip().strip("[]")
        if version and version.lower() != "unreleased":
            versions.add(version)
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    match = TAG_PATTERN.fullmatch(args.tag)
    if match is None:
        print(f"unsupported release tag format: {args.tag}", file=sys.stderr)
        return 1

    version = match.group("version")
    project_version = tomllib.loads((repo_root / "pyproject.toml").read_text())[
        "project"
    ]["version"]
    if version != project_version:
        print(
            f"tag version {version} does not match pyproject version {project_version}",
            file=sys.stderr,
        )
        return 1

    changelog_path = repo_root / "CHANGELOG.md"
    if not changelog_path.is_file():
        print("CHANGELOG.md is required for releases", file=sys.stderr)
        return 1

    versions = _extract_versions(changelog_path)
    if version not in versions:
        print(
            f"CHANGELOG.md does not contain a release section for version {version}",
            file=sys.stderr,
        )
        return 1

    print(f"release metadata validated for {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
