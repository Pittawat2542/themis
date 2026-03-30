from __future__ import annotations

import subprocess
import sys


def test_cli_help_lists_phase5_command_groups() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "themis.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for command in (
        "quick-eval",
        "run",
        "submit",
        "resume",
        "estimate",
        "report",
        "quickcheck",
        "compare",
        "export",
        "init",
        "worker",
        "batch",
    ):
        assert command in result.stdout
