from __future__ import annotations

from themis.core.read_models import BenchmarkResult, BenchmarkScoreRow
from themis.core.stats import ComparisonSummary, StatsEngine, StatsSummary


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


def test_stats_engine_summarizes_rows_by_metric() -> None:
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

    summary = StatsEngine().summarize(benchmark_result)

    assert isinstance(summary, StatsSummary)
    assert summary.run_id == "run-1"
    assert summary.total_cases == 2
    assert summary.completed_cases == 2
    assert summary.failed_cases == 0
    assert [metric.model_dump() for metric in summary.metrics] == [
        {
            "metric_id": "accuracy",
            "count": 2,
            "mean": 0.5,
            "min": 0.0,
            "max": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
        },
        {
            "metric_id": "f1",
            "count": 2,
            "mean": 0.5,
            "min": 0.25,
            "max": 0.75,
            "ci_lower": 0.25,
            "ci_upper": 0.75,
        },
    ]


def test_stats_engine_compare_aligns_rows_by_case_and_metric() -> None:
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

    comparison = StatsEngine().compare(baseline, contender)

    assert isinstance(comparison, ComparisonSummary)
    assert comparison.baseline_run_id == "baseline"
    assert comparison.candidate_run_id == "contender"
    assert [metric.model_dump() for metric in comparison.metrics] == [
        {
            "metric_id": "accuracy",
            "pairs": 2,
            "wins": 1,
            "losses": 0,
            "ties": 1,
            "mean_delta": 0.5,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "p_value": 1.0,
            "effect_size": 0.7071067812,
        },
        {
            "metric_id": "f1",
            "pairs": 1,
            "wins": 1,
            "losses": 0,
            "ties": 0,
            "mean_delta": 0.4,
            "ci_lower": 0.4,
            "ci_upper": 0.4,
            "p_value": 1.0,
            "effect_size": 0.0,
        },
    ]


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

    summary = StatsEngine().summarize(benchmark_result)
    metric = summary.metrics[0]

    assert metric.mean == 0.5
    assert metric.ci_lower == 0.0
    assert metric.ci_upper == 1.0


def test_stats_engine_ignores_error_and_missing_score_values() -> None:
    benchmark_result = _benchmark_result(
        "run-3",
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
                value=None,
                candidate_id="candidate-b",
            ),
            BenchmarkScoreRow(
                case_id="case-3",
                metric_id="accuracy",
                value=0.0,
                outcome="error",
                candidate_id="candidate-c",
            ),
        ],
        total_cases=3,
        completed_cases=2,
        failed_cases=1,
    )

    summary = StatsEngine().summarize(benchmark_result)

    assert [metric.model_dump() for metric in summary.metrics] == [
        {
            "metric_id": "accuracy",
            "count": 1,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "ci_lower": 1.0,
            "ci_upper": 1.0,
        }
    ]


def test_stats_engine_effect_size_is_zero_for_equal_paired_deltas() -> None:
    baseline = _benchmark_result(
        "baseline",
        [
            BenchmarkScoreRow(
                case_id="case-1",
                metric_id="accuracy",
                value=0.2,
                candidate_id="candidate-a",
            ),
            BenchmarkScoreRow(
                case_id="case-2",
                metric_id="accuracy",
                value=0.4,
                candidate_id="candidate-b",
            ),
        ],
    )
    contender = _benchmark_result(
        "contender",
        [
            BenchmarkScoreRow(
                case_id="case-1",
                metric_id="accuracy",
                value=0.5,
                candidate_id="candidate-a",
            ),
            BenchmarkScoreRow(
                case_id="case-2",
                metric_id="accuracy",
                value=0.7,
                candidate_id="candidate-b",
            ),
        ],
    )

    comparison = StatsEngine().compare(baseline, contender)

    assert comparison.metrics[0].mean_delta == 0.3
    assert comparison.metrics[0].effect_size == 0.0
