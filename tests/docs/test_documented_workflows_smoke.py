from __future__ import annotations

import subprocess
import sys


def test_examples_simple_comparison_workflow_runs():
    result = subprocess.run(
        [sys.executable, "examples/04_comparison.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
