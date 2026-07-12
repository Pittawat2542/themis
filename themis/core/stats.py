"""Statistical summaries over projection-backed benchmark results."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from enum import StrEnum
from math import comb, sqrt

from themis.core.base import FrozenModel
from themis.core.read_models import BenchmarkResult


class MetricSummary(FrozenModel):
    """Summary statistics for one metric in a benchmark result."""

    metric_id: str
    count: int
    mean: float
    min: float
    max: float
    ci_lower: float
    ci_upper: float


class StatsSummary(FrozenModel):
    """Run-level statistical summary over benchmark metric rows."""

    run_id: str
    total_cases: int
    completed_cases: int
    failed_cases: int
    metrics: list[MetricSummary]


class MetricComparison(FrozenModel):
    """Paired comparison statistics for one metric across two runs."""

    metric_id: str
    direction: MetricDirection
    baseline_count: int
    candidate_count: int
    pairs: int
    unpaired_baseline: int
    unpaired_candidate: int
    wins: int
    losses: int
    ties: int
    mean_delta: float
    mean_improvement: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size: float


class ComparisonSummary(FrozenModel):
    """Run-level paired comparison between a baseline and candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    metrics: list[MetricComparison]


class MetricDirection(StrEnum):
    """Declares how a metric value maps to quality."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class StatsEngine:
    """Dependency-free statistics over projection-backed benchmark results."""

    def summarize(self, benchmark_result: BenchmarkResult) -> StatsSummary:
        """Return typed summary statistics for one benchmark result."""

        metric_values: dict[str, list[float]] = defaultdict(list)
        for row in benchmark_result.score_rows:
            if row.value is None or row.outcome == "error":
                continue
            metric_values[row.metric_id].append(float(row.value))

        return StatsSummary(
            run_id=benchmark_result.run_id,
            total_cases=benchmark_result.total_cases,
            completed_cases=benchmark_result.completed_cases,
            failed_cases=benchmark_result.failed_cases,
            metrics=[
                MetricSummary(
                    metric_id=metric_id,
                    count=len(values),
                    mean=_rounded(sum(values) / len(values)),
                    min=_rounded(min(values)),
                    max=_rounded(max(values)),
                    ci_lower=_rounded(_bootstrap_mean_ci(values)[0]),
                    ci_upper=_rounded(_bootstrap_mean_ci(values)[1]),
                )
                for metric_id, values in sorted(metric_values.items())
                if values
            ],
        )

    def compare(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
        *,
        directions: Mapping[str, MetricDirection | str] | None = None,
    ) -> ComparisonSummary:
        """Return a paired comparison with explicit per-metric direction."""

        baseline_rows = {
            (_comparison_case_key(row), row.metric_id): float(row.value)
            for row in baseline.score_rows
            if row.value is not None and row.outcome != "error"
        }
        candidate_rows = {
            (_comparison_case_key(row), row.metric_id): float(row.value)
            for row in candidate.score_rows
            if row.value is not None and row.outcome != "error"
        }

        metric_deltas: dict[str, list[float]] = defaultdict(list)
        for key, baseline_value in baseline_rows.items():
            candidate_value = candidate_rows.get(key)
            if candidate_value is None:
                continue
            _, metric_id = key
            metric_deltas[metric_id].append(candidate_value - baseline_value)

        resolved_directions = {
            metric_id: MetricDirection(direction)
            for metric_id, direction in dict(directions or {}).items()
        }
        baseline_counts = _metric_counts(baseline_rows)
        candidate_counts = _metric_counts(candidate_rows)

        return ComparisonSummary(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            metrics=[
                MetricComparison(
                    metric_id=metric_id,
                    direction=resolved_directions.get(
                        metric_id, MetricDirection.HIGHER_IS_BETTER
                    ),
                    baseline_count=baseline_counts[metric_id],
                    candidate_count=candidate_counts[metric_id],
                    pairs=len(deltas),
                    unpaired_baseline=baseline_counts[metric_id] - len(deltas),
                    unpaired_candidate=candidate_counts[metric_id] - len(deltas),
                    wins=sum(
                        1
                        for delta in _improvements(
                            deltas, resolved_directions.get(metric_id)
                        )
                        if delta > 0
                    ),
                    losses=sum(
                        1
                        for delta in _improvements(
                            deltas, resolved_directions.get(metric_id)
                        )
                        if delta < 0
                    ),
                    ties=sum(1 for delta in deltas if delta == 0),
                    mean_delta=_rounded(sum(deltas) / len(deltas)),
                    mean_improvement=_rounded(
                        sum(_improvements(deltas, resolved_directions.get(metric_id)))
                        / len(deltas)
                    ),
                    ci_lower=_rounded(_bootstrap_mean_ci(deltas)[0]),
                    ci_upper=_rounded(_bootstrap_mean_ci(deltas)[1]),
                    p_value=_rounded(_paired_sign_test_p_value(deltas)),
                    effect_size=_rounded(
                        _effect_size(
                            _improvements(deltas, resolved_directions.get(metric_id))
                        )
                    ),
                )
                for metric_id, deltas in sorted(metric_deltas.items())
                if deltas
            ],
        )


def _metric_counts(rows: Mapping[tuple[str, str], float]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _, metric_id in rows:
        counts[metric_id] += 1
    return counts


def _improvements(
    deltas: list[float], direction: MetricDirection | None
) -> list[float]:
    multiplier = -1.0 if direction is MetricDirection.LOWER_IS_BETTER else 1.0
    return [delta * multiplier for delta in deltas]


def _rounded(value: float) -> float:
    return round(value, 10)


def _comparison_case_key(row: object) -> str:
    case_key = getattr(row, "case_key", None)
    if isinstance(case_key, str) and case_key:
        return case_key
    dataset_id = getattr(row, "dataset_id", None)
    case_id = getattr(row, "case_id")
    if isinstance(dataset_id, str) and dataset_id:
        return f"{len(dataset_id)}:{dataset_id}:{case_id}"
    return str(case_id)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    sample_size = len(values)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        means.append(sum(sample) / sample_size)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = int(alpha * (resamples - 1))
    upper_index = int((1.0 - alpha) * (resamples - 1))
    return means[lower_index], means[upper_index]


def _paired_sign_test_p_value(deltas: list[float]) -> float:
    non_zero_deltas = [delta for delta in deltas if delta != 0]
    pair_count = len(non_zero_deltas)
    if pair_count == 0:
        return 1.0
    positive = sum(1 for delta in non_zero_deltas if delta > 0)
    negative = pair_count - positive
    tail = min(positive, negative)
    probability = sum(comb(pair_count, k) for k in range(tail + 1)) / (2**pair_count)
    return min(1.0, 2.0 * probability)


def _effect_size(deltas: list[float]) -> float:
    if len(deltas) <= 1:
        return 0.0
    mean_delta = sum(deltas) / len(deltas)
    variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (len(deltas) - 1)
    if variance <= 1e-24:
        return 0.0
    return mean_delta / sqrt(variance)
