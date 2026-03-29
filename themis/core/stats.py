"""Statistical summaries over projection-backed benchmark results."""

from __future__ import annotations

from collections import defaultdict

from themis.core.read_models import BenchmarkResult


class StatsEngine:
    def aggregate(self, benchmark_result: BenchmarkResult) -> dict[str, object]:
        metric_values: dict[str, list[float]] = defaultdict(list)
        for row in benchmark_result.score_rows:
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
        }
        candidate_rows = {
            (row.case_id, row.metric_id): float(row.value)
            for row in candidate.score_rows
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
                }
                for metric_id, deltas in sorted(metric_deltas.items())
                if deltas
            },
        }


def _rounded(value: float) -> float:
    return round(value, 10)
