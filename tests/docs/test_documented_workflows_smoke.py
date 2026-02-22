from __future__ import annotations

import subprocess
import sys


import os


def test_examples_simple_comparison_workflow_runs():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "examples/04_comparison.py"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert result.returncode == 0, result.stderr or result.stdout
