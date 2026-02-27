"""Tests for import performance."""

import subprocess
import sys


def test_import_time_under_threshold():
    """Import time should be under 2 seconds (target: <1s, gate: <2s)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; t=time.perf_counter(); import themis; "
            "print(time.perf_counter()-t)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    elapsed = float(result.stdout.strip())
    assert elapsed < 2.0, f"Import took {elapsed:.2f}s, expected < 2.0s"
