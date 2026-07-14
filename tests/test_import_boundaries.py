from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_IMPORT = re.compile(r"^\s*(?:from\s+themis\.core|import\s+themis\.core)")


def test_user_facing_sources_do_not_import_themis_core() -> None:
    sources = [ROOT / "README.md"]
    for directory in (ROOT / "docs", ROOT / "examples"):
        sources.extend(directory.rglob("*.md"))
        sources.extend(directory.rglob("*.py"))

    violations = [
        f"{path.relative_to(ROOT)}:{line_number}"
        for path in sources
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if PRIVATE_IMPORT.match(line)
    ]

    assert violations == [], "private imports in user-facing sources: " + ", ".join(
        violations
    )
