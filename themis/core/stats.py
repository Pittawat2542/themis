"""Statistical summaries over projection-backed benchmark results."""

from __future__ import annotations

import random
from collections import defaultdict
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
    pairs: int
    wins: int
    losses: int
    ties: int
    mean_delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size: float


class ComparisonSummary(FrozenModel):
    """Run-level paired comparison between a baseline and candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    metrics: list[MetricComparison]


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
    ) -> ComparisonSummary:
        """Return a typed paired comparison between two benchmark results."""

        baseline_rows = {
            (row.case_id, row.metric_id): float(row.value)
            for row in baseline.score_rows
            if row.value is not None and row.outcome != "error"
        }
        candidate_rows = {
            (row.case_id, row.metric_id): float(row.value)
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

        return ComparisonSummary(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            metrics=[
                MetricComparison(
                    metric_id=metric_id,
                    pairs=len(deltas),
                    wins=sum(1 for delta in deltas if delta > 0),
                    losses=sum(1 for delta in deltas if delta < 0),
                    ties=sum(1 for delta in deltas if delta == 0),
                    mean_delta=_rounded(sum(deltas) / len(deltas)),
                    ci_lower=_rounded(_bootstrap_mean_ci(deltas)[0]),
                    ci_upper=_rounded(_bootstrap_mean_ci(deltas)[1]),
                    p_value=_rounded(_paired_sign_test_p_value(deltas)),
                    effect_size=_rounded(_effect_size(deltas)),
                )
                for metric_id, deltas in sorted(metric_deltas.items())
                if deltas
            ],
        )


def _rounded(value: float) -> float:
    return round(value, 10)


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
