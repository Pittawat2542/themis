"""Statistical summaries over projection-backed benchmark results."""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb

from themis.core.read_models import BenchmarkResult


class StatsEngine:
    def aggregate(self, benchmark_result: BenchmarkResult) -> dict[str, object]:
        metric_values: dict[str, list[float]] = defaultdict(list)
        for row in benchmark_result.score_rows:
            if row.value is None or row.outcome == "error":
                continue
            metric_values[row.metric_id].append(float(row.value))

        return {
            "run_id": benchmark_result.run_id,
            "total_cases": benchmark_result.total_cases,
            "completed_cases": benchmark_result.completed_cases,
            "failed_cases": benchmark_result.failed_cases,
            "metrics": {
                metric_id: {
                    "count": len(values),
                    "mean": _rounded(sum(values) / len(values)),
                    "min": _rounded(min(values)),
                    "max": _rounded(max(values)),
                    "ci_lower": _rounded(_bootstrap_mean_ci(values)[0]),
                    "ci_upper": _rounded(_bootstrap_mean_ci(values)[1]),
                }
                for metric_id, values in sorted(metric_values.items())
                if values
            },
        }

    def paired_compare(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
    ) -> dict[str, object]:
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

        return {
            "baseline_run_id": baseline.run_id,
            "candidate_run_id": candidate.run_id,
            "metrics": {
                metric_id: {
                    "pairs": len(deltas),
                    "wins": sum(1 for delta in deltas if delta > 0),
                    "losses": sum(1 for delta in deltas if delta < 0),
                    "ties": sum(1 for delta in deltas if delta == 0),
                    "mean_delta": _rounded(sum(deltas) / len(deltas)),
                    "ci_lower": _rounded(_bootstrap_mean_ci(deltas)[0]),
                    "ci_upper": _rounded(_bootstrap_mean_ci(deltas)[1]),
                    "p_value": _rounded(_paired_sign_test_p_value(deltas)),
                }
                for metric_id, deltas in sorted(metric_deltas.items())
                if deltas
            },
        }


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
