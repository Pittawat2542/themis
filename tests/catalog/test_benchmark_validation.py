from __future__ import annotations

from themis.catalog import validate_benchmark


def test_validate_benchmark_passes_ready_code_execution_benchmarks() -> None:
    result = validate_benchmark("livecodebench")

    assert result.benchmark_id == "livecodebench"
    assert result.support_tier == "ready"
    assert result.checks["load"].status == "passed"
    assert result.checks["materialize"].status == "passed"
    assert result.checks["score_smoke"].status == "passed"


def test_validate_benchmark_passes_humaneval_benchmarks() -> None:
    result = validate_benchmark("humaneval")
    plus_result = validate_benchmark("humaneval_plus")

    assert result.benchmark_id == "humaneval"
    assert result.support_tier == "ready"
    assert result.checks["load"].status == "passed"
    assert result.checks["materialize"].status == "passed"
    assert result.checks["score_smoke"].status == "passed"
    assert plus_result.benchmark_id == "humaneval_plus"
    assert plus_result.support_tier == "ready"
    assert plus_result.checks["score_smoke"].status == "passed"
