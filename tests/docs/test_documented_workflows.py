from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIRS = [
    PROJECT_ROOT / ".cache",
    PROJECT_ROOT / ".themis_cache",
]


def _reset_example_state() -> None:
    for cache_dir in CACHE_DIRS:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _run_command(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _run_legacy_quickcheck(*args: str) -> str:
    return _run_command(
        sys.executable,
        "-c",
        "from themis.cli.quickcheck import main; raise SystemExit(main())",
        *args,
    )


def _run_parent_cli(*args: str) -> str:
    return _run_command(
        sys.executable,
        "-c",
        "from themis.cli.main import main; raise SystemExit(main())",
        *args,
    )


def test_project_file_example_formats_display_path_with_forward_slashes() -> None:
    module_globals = runpy.run_path(
        str(PROJECT_ROOT / "examples/02_project_file.py"),
        run_name="themis_example_02_project_file",
    )

    format_display_path = module_globals["_format_display_path"]

    assert (
        format_display_path(
            PureWindowsPath(
                ".cache/themis-examples/02-project-file-config/project.toml"
            )
        )
        == ".cache/themis-examples/02-project-file-config/project.toml"
    )


def test_documented_example_outputs_match_runnable_examples() -> None:
    _reset_example_state()

    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()
    hello_world_tutorial = (PROJECT_ROOT / "docs/tutorials/hello-world.md").read_text()
    hello_world_output = _run_command(sys.executable, "examples/01_hello_world.py")

    for expected_line in [
        "Stored SQLite database: .cache/themis-examples/01-hello-world/themis.sqlite3",
        "item-1: exact_match=1.0",
        "item-2: exact_match=1.0",
    ]:
        assert expected_line in hello_world_output
        assert expected_line in quick_start
        assert expected_line in hello_world_tutorial

    project_file_tutorial = (
        PROJECT_ROOT / "docs/tutorials/project-files.md"
    ).read_text()
    project_file_output = _run_command(sys.executable, "examples/02_project_file.py")

    for expected_line in [
        "Loaded project file: .cache/themis-examples/02-project-file-config/project.toml",
        "Trial hashes: 4842354de2b4, defddb7e8085",
    ]:
        assert expected_line in project_file_output
        assert expected_line in project_file_tutorial

    compare_guide = (PROJECT_ROOT / "docs/guides/compare-and-export.md").read_text()
    compare_output = _run_command(sys.executable, "examples/04_compare_models.py")

    for expected_line in [
        "delta_mean= 0.5 adjusted_p_value= 0.25 pairs= 6",
        "Report written to: .cache/themis-examples/04-compare-models/report.md",
    ]:
        assert expected_line in compare_output
    assert "delta_mean= 0.5 adjusted_p_value= 0.25 pairs= 6" in compare_guide


def test_documented_quickcheck_outputs_match_cli() -> None:
    _reset_example_state()
    _run_command(sys.executable, "examples/01_hello_world.py")

    db_path = ".cache/themis-examples/01-hello-world/themis.sqlite3"
    quickcheck_guide = (PROJECT_ROOT / "docs/guides/quickcheck.md").read_text()

    legacy_scores = _run_legacy_quickcheck(
        "scores", "--db", db_path, "--metric", "exact_match"
    )
    parent_scores = _run_parent_cli(
        "quickcheck",
        "scores",
        "--db",
        db_path,
        "--metric",
        "exact_match",
    )
    assert legacy_scores == parent_scores
    assert (
        "ev:fc7ad3e8b3e2\tdemo-model\tarithmetic\texact_match\t1.0000\t2"
        in legacy_scores
    )
    assert (
        "ev:fc7ad3e8b3e2\tdemo-model\tarithmetic\texact_match\t1.0000\t2"
        in quickcheck_guide
    )

    latency_output = _run_legacy_quickcheck("latency", "--db", db_path)
    assert (
        "count=2 latency_ms(avg=2.00,p50=2.00,p95=2.00) "
        "tokens_in(avg=n/a) tokens_out(avg=n/a)"
    ) in latency_output
    assert (
        "count=2 latency_ms(avg=2.00,p50=2.00,p95=2.00) "
        "tokens_in(avg=n/a) tokens_out(avg=n/a)"
    ) in quickcheck_guide

    failures_output = _run_legacy_quickcheck(
        "failures", "--db", db_path, "--limit", "20"
    )
    assert failures_output == ""
    assert (
        "For a fully successful run like the hello-world example, `failures` prints no"
        in quickcheck_guide
    )


def test_documented_cli_help_matches_current_output() -> None:
    installation = (PROJECT_ROOT / "docs/installation-setup/index.md").read_text()
    config_reports = (PROJECT_ROOT / "docs/guides/config-reports.md").read_text()

    quickcheck_help = _run_command(
        sys.executable,
        "-c",
        "from themis.cli.quickcheck import main; raise SystemExit(main())",
        "--help",
    )
    assert (
        "usage: themis-quickcheck [-h] {failures,scores,latency} ..." in quickcheck_help
    )
    assert "usage: themis-quickcheck [-h] {failures,scores,latency} ..." in installation

    report_help = _run_parent_cli("report", "--help")
    for expected_line in [
        "usage: themis report [-h] (--factory FACTORY | --project-file PROJECT_FILE)",
        "[--run-id RUN_ID] [--format {json,yaml,markdown,latex}]",
        "[--verbosity {default,full}] [--output OUTPUT]",
    ]:
        assert expected_line in report_help
        assert expected_line in config_reports
