from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TAG_PATTERN = re.compile(r"^v(?P<version>[0-9]+(?:\.[0-9]+){2}(?:[A-Za-z0-9.-]+)?)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changelog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    match = TAG_PATTERN.fullmatch(args.tag)
    if match is None:
        print(f"unsupported release tag format: {args.tag}", file=sys.stderr)
        return 1

    version = match.group("version")
    changelog_lines = Path(args.changelog).read_text(encoding="utf-8").splitlines()
    start: int | None = None
    end = len(changelog_lines)

    for index, line in enumerate(changelog_lines):
        if not line.startswith("## "):
            continue
        heading = line[3:].strip()
        heading_version = heading.split(" - ", 1)[0].strip().strip("[]")
        if heading_version == version:
            start = index
            continue
        if start is not None:
            end = index
            break

    if start is None:
        print(f"could not find changelog section for {version}", file=sys.stderr)
        return 1

    body = "\n".join(changelog_lines[start:end]).strip()
    Path(args.output).write_text(body + "\n", encoding="utf-8")
    print(f"wrote release notes for {args.tag} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
