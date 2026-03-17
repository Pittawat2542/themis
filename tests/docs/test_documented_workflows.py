from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_CMD_TIMEOUT = 120
EXAMPLE_CACHE_ROOT = PROJECT_ROOT / ".cache" / "themis-examples"


def _reset_example_state() -> None:
    shutil.rmtree(EXAMPLE_CACHE_ROOT, ignore_errors=True)


def _run_command(*args: str, timeout: int = DOCS_CMD_TIMEOUT) -> str:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _run_quickcheck(*args: str) -> str:
    return _run_command(
        sys.executable,
        "-c",
        "from themis.cli.quickcheck import main; raise SystemExit(main())",
        *args,
    )


def test_hello_world_docs_match_runnable_example() -> None:
    _reset_example_state()

    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()
    tutorial = (PROJECT_ROOT / "docs/tutorials/hello-world.md").read_text()
    output = _run_command(sys.executable, "examples/01_hello_world.py").strip()

    assert output in quick_start
    assert output in tutorial


def test_compare_docs_match_runnable_example() -> None:
    _reset_example_state()

    compare_guide = (PROJECT_ROOT / "docs/guides/compare-and-export.md").read_text()
    output = _run_command(sys.executable, "examples/04_compare_models.py")

    expected = (
        "[{'slice_id': 'qa', 'metric_id': 'exact_match', 'baseline_model_id': "
        "'baseline', 'treatment_model_id': 'candidate', 'pair_count': 4, "
        "'baseline_mean': 0.5, 'treatment_mean': 1.0, 'delta_mean': 0.5, "
        "'p_value': 0.5, 'adjusted_p_value': 0.5, 'adjustment_method': "
        "<PValueCorrection.NONE: 'none'>, 'ci_lower': 0.0, 'ci_upper': 1.0, "
        "'ci_level': 0.95, 'method': 'bootstrap_BCa_wilcoxon'}]"
    )
    assert expected in output
    assert expected in compare_guide


def test_quickcheck_docs_match_current_cli() -> None:
    _reset_example_state()
    _run_command(sys.executable, "examples/01_hello_world.py")
    _run_command(sys.executable, "examples/04_compare_models.py")

    quickcheck_guide = (PROJECT_ROOT / "docs/guides/quickcheck.md").read_text()

    hello_scores = _run_quickcheck(
        "scores",
        "--db",
        ".cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3",
        "--metric",
        "exact_match",
    ).strip()
    assert hello_scores in quickcheck_guide

    slice_scores = _run_quickcheck(
        "scores",
        "--db",
        ".cache/themis-examples/04-compare-models-benchmark-first/themis.sqlite3",
        "--metric",
        "exact_match",
        "--slice",
        "qa",
    )
    assert "baseline\tqa\texact_match\t0.5000\t4" in slice_scores
    assert "--slice qa" in quickcheck_guide
    assert "--dimension source=synthetic" in quickcheck_guide
