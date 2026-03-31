from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_inventory_script_reports_public_surface() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/docs/build_inventory.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert sorted(payload) == [
        "benchmarks",
        "builtin_components",
        "catalog_exports",
        "cli_commands",
        "docs_destinations",
        "public_exports",
        "required_topics",
    ]
    assert "Experiment" in payload["public_exports"]
    assert "load" in payload["catalog_exports"]
    assert "quick-eval benchmark" in payload["cli_commands"]
    assert "builtin/exact_match" in payload["builtin_components"]
    assert "mmlu_pro" in payload["benchmarks"]
    assert payload["docs_destinations"]["glossary"] == "docs/glossary.md"
    assert "LifecycleSubscriber" in payload["required_topics"]["observability"]
