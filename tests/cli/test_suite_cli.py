from __future__ import annotations

import json

from tests.cli.helpers import run_cli


def test_suite_cli_lists_and_inspects_builtin_suites() -> None:
    listed = run_cli("suite", "list")
    inspected = run_cli("suite", "inspect", "math-core")

    assert listed.returncode == 0
    assert "math-core" in listed.stdout
    assert inspected.returncode == 0
    payload = json.loads(inspected.stdout)
    assert payload["suite_id"] == "math-core"
    assert payload["items"]


def test_suite_cli_runs_suite_with_memory_store() -> None:
    result = run_cli("suite", "run", "math-core")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["suite_id"] == "math-core"
    assert payload["run_ids"]
