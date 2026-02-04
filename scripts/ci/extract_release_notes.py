#!/usr/bin/env python3
"""Extract release notes for a version from CHANGELOG.md."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADER_RE = re.compile(r"^## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", required=True, help="Release tag (for example: v1.2.3)"
    )
    parser.add_argument(
        "--changelog", required=True, type=Path, help="Path to CHANGELOG"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output markdown file"
    )
    return parser.parse_args()


def normalize_version(tag: str) -> str:
    value = tag.strip()
    if value.startswith("refs/tags/"):
        value = value.removeprefix("refs/tags/")
    if value.startswith("v"):
        value = value[1:]
    return value


def extract_release_section(changelog_lines: list[str], version: str) -> list[str]:
    start_idx: int | None = None
    end_idx: int | None = None

    for idx, line in enumerate(changelog_lines):
        match = HEADER_RE.match(line.strip())
        if match and match.group("version") == version:
            start_idx = idx
            continue
        if start_idx is not None and match:
            end_idx = idx
            break

    if start_idx is None:
        raise ValueError(f"No changelog entry found for version {version}")

    section = changelog_lines[start_idx : end_idx or len(changelog_lines)]
    return section


def main() -> int:
    args = parse_args()
    version = normalize_version(args.tag)

    if not args.changelog.exists():
        print(f"Changelog not found: {args.changelog}", file=sys.stderr)
        return 1

    lines = args.changelog.read_text(encoding="utf-8").splitlines()

    try:
        section = extract_release_section(lines, version)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    content = "\n".join(section).strip()
    args.output.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote release notes for {version} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
