from __future__ import annotations

import subprocess
import sys


def test_cli_help_lists_command_groups() -> None:
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


def test_worker_run_help_is_single_shot_without_once_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "themis.cli", "worker", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--queue-root" in result.stdout
    assert "--once" not in result.stdout
