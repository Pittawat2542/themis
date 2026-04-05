from __future__ import annotations

from typing import cast

from themis.core.read_models import BenchmarkResult, BenchmarkScoreRow
from themis.core.stats import StatsEngine


def _benchmark_result(
    run_id: str,
    rows: list[BenchmarkScoreRow],
    *,
    total_cases: int = 2,
    completed_cases: int = 2,
    failed_cases: int = 0,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=run_id,
        dataset_ids=["dataset-1"],
        metric_ids=sorted({row.metric_id for row in rows}),
        total_cases=total_cases,
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        score_rows=rows,
        metric_means={},
    )


def test_stats_engine_aggregates_rows_by_metric() -> None:
    benchmark_result = _benchmark_result(
        "run-1",
        [
            BenchmarkScoreRow(
                case_id="case-1",
                metric_id="accuracy",
                value=1.0,
                candidate_id="candidate-a",
            ),
            BenchmarkScoreRow(
                case_id="case-2",
                metric_id="accuracy",
                value=0.0,
                candidate_id="candidate-b",
            ),
            BenchmarkScoreRow(
                case_id="case-1", metric_id="f1", value=0.25, candidate_id="candidate-a"
            ),
            BenchmarkScoreRow(
                case_id="case-2", metric_id="f1", value=0.75, candidate_id="candidate-b"
            ),
        ],
    )

    summary = StatsEngine().aggregate(benchmark_result)
    metrics = cast(dict[str, dict[str, float | int]], summary["metrics"])

    assert summary["run_id"] == "run-1"
    assert summary["total_cases"] == 2
    assert summary["completed_cases"] == 2
    assert summary["failed_cases"] == 0
    assert metrics["accuracy"] == {
        "count": 2,
        "mean": 0.5,
        "min": 0.0,
        "max": 1.0,
        "ci_lower": 0.0,
        "ci_upper": 1.0,
    }
    assert metrics["f1"] == {
        "count": 2,
        "mean": 0.5,
        "min": 0.25,
        "max": 0.75,
        "ci_lower": 0.25,
        "ci_upper": 0.75,
    }


def test_stats_engine_paired_compare_aligns_rows_by_case_and_metric() -> None:
    baseline = _benchmark_result(
        "baseline",
        [
            BenchmarkScoreRow(
                case_id="case-1",
                metric_id="accuracy",
                value=0.0,
                candidate_id="candidate-a",
            ),
            BenchmarkScoreRow(
                case_id="case-2",
                metric_id="accuracy",
                value=0.5,
                candidate_id="candidate-b",
            ),
            BenchmarkScoreRow(
                case_id="case-1", metric_id="f1", value=0.2, candidate_id="candidate-a"
            ),
        ],
    )
    contender = _benchmark_result(
        "contender",
        [
            BenchmarkScoreRow(
                case_id="case-1",
                metric_id="accuracy",
                value=1.0,
                candidate_id="candidate-a",
            ),
            BenchmarkScoreRow(
                case_id="case-2",
                metric_id="accuracy",
                value=0.5,
                candidate_id="candidate-b",
            ),
            BenchmarkScoreRow(
                case_id="case-1", metric_id="f1", value=0.6, candidate_id="candidate-a"
            ),
            BenchmarkScoreRow(
                case_id="case-2", metric_id="f1", value=0.9, candidate_id="candidate-b"
            ),
        ],
    )

    comparison = StatsEngine().paired_compare(baseline, contender)
    metrics = cast(dict[str, dict[str, float | int]], comparison["metrics"])

    assert comparison["baseline_run_id"] == "baseline"
    assert comparison["candidate_run_id"] == "contender"
    assert metrics["accuracy"] == {
        "pairs": 2,
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "mean_delta": 0.5,
        "ci_lower": 0.0,
        "ci_upper": 1.0,
        "p_value": 1.0,
    }
    assert metrics["f1"] == {
        "pairs": 1,
        "wins": 1,
        "losses": 0,
        "ties": 0,
        "mean_delta": 0.4,
        "ci_lower": 0.4,
        "ci_upper": 0.4,
        "p_value": 1.0,
    }


def test_stats_engine_reports_confidence_intervals_for_metric_means() -> None:
    benchmark_result = _benchmark_result(
        "run-2",
        [
            BenchmarkScoreRow(
                case_id="case-1",
                metric_id="accuracy",
                value=1.0,
                candidate_id="candidate-a",
            ),
            BenchmarkScoreRow(
                case_id="case-2",
                metric_id="accuracy",
                value=0.0,
                candidate_id="candidate-b",
            ),
            BenchmarkScoreRow(
                case_id="case-3",
                metric_id="accuracy",
                value=1.0,
                candidate_id="candidate-c",
            ),
            BenchmarkScoreRow(
                case_id="case-4",
                metric_id="accuracy",
                value=0.0,
                candidate_id="candidate-d",
            ),
        ],
        total_cases=4,
        completed_cases=4,
    )

    summary = StatsEngine().aggregate(benchmark_result)
    metric = cast(dict[str, float | int], cast(dict[str, object], summary["metrics"])["accuracy"])

    assert metric["mean"] == 0.5
    assert metric["ci_lower"] == 0.0
    assert metric["ci_upper"] == 1.0
