from __future__ import annotations

from themis.catalog import validate_benchmark


def test_validate_benchmark_requires_sandbox_for_code_scoring(
    catalog_fixture_loader: None,
) -> None:
    result = validate_benchmark("livecodebench")

    assert result.benchmark_id == "livecodebench"
    assert result.support_tier == "ready"
    assert result.checks["load"].status == "passed"
    assert result.checks["materialize"].status == "passed"
    assert result.checks["score_smoke"].status == "skipped"
    assert "explicit sandbox executor" in (result.checks["score_smoke"].message or "")


def test_validate_humaneval_plus_requires_sandbox_for_code_scoring(
    catalog_fixture_loader: None,
) -> None:
    result = validate_benchmark("humaneval_plus")

    assert result.benchmark_id == "humaneval_plus"
    assert result.support_tier == "ready"
    assert result.checks["load"].status == "passed"
    assert result.checks["materialize"].status == "passed"
    assert result.checks["score_smoke"].status == "skipped"
