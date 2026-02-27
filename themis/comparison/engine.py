"""Comparison engine for analyzing multiple experiment runs.

This module provides the main ComparisonEngine class that orchestrates
loading runs, computing statistics, and generating comparison reports.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from themis.comparison import reports
from themis.evaluation.statistics import comparison_tests as statistics
from themis.evaluation.statistics.comparison_tests import StatisticalTest
from themis.evaluation.statistics import holm_bonferroni
from themis.exceptions import ConfigurationError, EvaluationError, MetricError
from themis.storage import ExperimentStorage


class ComparisonEngine:
    """Engine for comparing multiple experiment runs.

    This class loads experiment results from storage and performs
    pairwise comparisons across all metrics with statistical testing.
    """

    def __init__(
        self,
        *,
        storage: ExperimentStorage | None = None,
        storage_path: str | Path | None = None,
        statistical_test: StatisticalTest = StatisticalTest.BOOTSTRAP,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000,
        multiple_comparison_correction: str | None = "holm-bonferroni",
    ):
        """Initialize comparison engine.

        Args:
            storage: Experiment storage instance
            storage_path: Path to storage (if storage not provided)
            statistical_test: Type of statistical test to use
            alpha: Significance level for tests
            n_bootstrap: Number of bootstrap iterations
            n_permutations: Number of permutations for permutation test
            multiple_comparison_correction: Correction policy for multiple
                comparisons. Use "holm-bonferroni" (default) or None.
        """
        if storage is None and storage_path is None:
            raise ConfigurationError("Either storage or storage_path must be provided")

        self._storage = storage or ExperimentStorage(storage_path)
        self._statistical_test = statistical_test
        self._alpha = alpha
        self._n_bootstrap = n_bootstrap
        self._n_permutations = n_permutations
        self._multiple_comparison_correction = multiple_comparison_correction

    def compare_runs(
        self,
        run_ids: Sequence[str],
        *,
        metrics: Sequence[str] | None = None,
        statistical_test: StatisticalTest | None = None,
    ) -> reports.ComparisonReport:
        """Compare multiple runs across specified metrics.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare (None = all available)
            statistical_test: Override default statistical test

        Returns:
            ComparisonReport with all comparisons and statistics

        Raises:
            ValueError: If fewer than 2 runs provided or runs not found
        """
        if len(run_ids) < 2:
            raise ConfigurationError("Need at least 2 runs to compare")
        effective_test = statistical_test or self._statistical_test

        # Load all runs
        run_data = {}
        for run_id in run_ids:
            try:
                data = self._load_run_metrics(run_id)
                run_data[run_id] = data
            except FileNotFoundError:
                raise EvaluationError(f"Run not found: {run_id}")

        # Determine metrics to compare
        if metrics is None:
            # Use all metrics that appear in all runs
            all_metrics = set(run_data[run_ids[0]].keys())
            for run_id in run_ids[1:]:
                all_metrics &= set(run_data[run_id].keys())
            metrics = sorted(all_metrics)

        if not metrics:
            raise MetricError("No common metrics found across all runs")

        # Perform pairwise comparisons
        pairwise_results = []
        for metric in metrics:
            for i, run_a in enumerate(run_ids):
                for run_b in run_ids[i + 1 :]:
                    result = self._compare_pair(
                        run_a,
                        run_b,
                        metric,
                        run_data[run_a][metric],
                        run_data[run_b][metric],
                        effective_test,
                    )
                    pairwise_results.append(result)

        self._apply_multiple_comparison_correction(pairwise_results)

        # Build win/loss matrices
        win_loss_matrices = {}
        for metric in metrics:
            matrix = self._build_win_loss_matrix(run_ids, metric, pairwise_results)
            win_loss_matrices[metric] = matrix

        # Determine best run per metric
        best_run_per_metric = {}
        for metric in metrics:
            # Find run with highest mean
            best_run = max(
                run_ids,
                key=lambda rid: (
                    sum(run_data[rid][metric].values()) / len(run_data[rid][metric])
                ),
            )
            best_run_per_metric[metric] = best_run

        # Determine overall best run (most wins across all metrics)
        overall_wins = {run_id: 0 for run_id in run_ids}
        for matrix in win_loss_matrices.values():
            for run_id in run_ids:
                overall_wins[run_id] += matrix.win_counts.get(run_id, 0)

        overall_best_run = max(overall_wins, key=overall_wins.get)

        return reports.ComparisonReport(
            run_ids=list(run_ids),
            metrics=list(metrics),
            pairwise_results=pairwise_results,
            win_loss_matrices=win_loss_matrices,
            best_run_per_metric=best_run_per_metric,
            overall_best_run=overall_best_run,
            metadata={
                "statistical_test": effective_test.value,
                "alpha": self._alpha,
                "n_runs": len(run_ids),
                "n_metrics": len(metrics),
                "multiple_comparison_correction": self._multiple_comparison_correction,
                "n_hypotheses_corrected": sum(
                    1
                    for result in pairwise_results
                    if _supports_p_value_correction(result)
                ),
            },
        )

    def _load_run_metrics(self, run_id: str) -> dict[str, dict[str, float]]:
        """Load all metric scores for a run.

        Returns:
            Dictionary mapping metric names to sample_id -> score mappings
        """
        # Load evaluation records from storage (returns dict of cache_key -> EvaluationRecord)
        eval_dict = self._storage.load_cached_evaluations(run_id)

        # Organize scores by metric
        metric_scores: dict[str, dict[str, list[float]]] = {}

        # eval_dict is a dict, so iterate over values
        for record in eval_dict.values():
            sample_id = record.sample_id
            if sample_id is None:
                continue
            for score_obj in record.scores:
                metric_name = score_obj.metric_name
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = {}

                score = score_obj.value
                metric_scores[metric_name].setdefault(sample_id, []).append(score)

        # Collapse duplicated sample IDs deterministically by mean to avoid
        # silently overwriting values during alignment.
        collapsed: dict[str, dict[str, float]] = {}
        for metric_name, sample_values in metric_scores.items():
            collapsed[metric_name] = {
                sample_id: (sum(values) / len(values))
                for sample_id, values in sample_values.items()
                if values
            }
        return collapsed

    def _compare_pair(
        self,
        run_a_id: str,
        run_b_id: str,
        metric_name: str,
        samples_a: dict[str, float],
        samples_b: dict[str, float],
        test_type: StatisticalTest,
    ) -> reports.ComparisonResult:
        """Compare two runs on a single metric.

        Args:
            run_a_id: First run identifier
            run_b_id: Second run identifier
            metric_name: Name of metric being compared
            samples_a: Scores for first run
            samples_b: Scores for second run
            test_type: Type of statistical test to perform

        Returns:
            ComparisonResult with comparison statistics
        """
        common_sample_ids = sorted(set(samples_a) & set(samples_b))
        if not common_sample_ids:
            raise MetricError(
                f"Cannot compare metric '{metric_name}' for runs '{run_a_id}' and "
                f"'{run_b_id}': no overlapping sample_ids."
            )

        aligned_a = [samples_a[sample_id] for sample_id in common_sample_ids]
        aligned_b = [samples_b[sample_id] for sample_id in common_sample_ids]

        # Calculate means
        mean_a = sum(aligned_a) / len(aligned_a)
        mean_b = sum(aligned_b) / len(aligned_b)

        # Calculate delta
        delta = mean_a - mean_b
        delta_percent = (delta / mean_b * 100) if mean_b != 0 else 0.0

        # Perform statistical test
        test_result = None
        if test_type == StatisticalTest.T_TEST:
            test_result = statistics.t_test(
                aligned_a, aligned_b, alpha=self._alpha, paired=True
            )
        elif test_type == StatisticalTest.BOOTSTRAP:
            test_result = statistics.bootstrap_confidence_interval(
                aligned_a,
                aligned_b,
                n_bootstrap=self._n_bootstrap,
                confidence_level=1 - self._alpha,
            )
        elif test_type == StatisticalTest.PERMUTATION:
            test_result = statistics.permutation_test(
                aligned_a,
                aligned_b,
                n_permutations=self._n_permutations,
                alpha=self._alpha,
            )

        # Determine winner
        if test_result and test_result.significant:
            winner = run_a_id if delta > 0 else run_b_id
        else:
            winner = "tie"

        return reports.ComparisonResult(
            metric_name=metric_name,
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            run_a_mean=mean_a,
            run_b_mean=mean_b,
            delta=delta,
            delta_percent=delta_percent,
            winner=winner,
            test_result=test_result,
            run_a_samples=aligned_a,
            run_b_samples=aligned_b,
        )

    def _apply_multiple_comparison_correction(
        self, pairwise_results: list[reports.ComparisonResult]
    ) -> None:
        """Apply configured multiple-comparison correction in-place."""
        if self._multiple_comparison_correction is None:
            return
        if self._multiple_comparison_correction != "holm-bonferroni":
            raise ConfigurationError(
                "Unsupported multiple comparison correction: "
                f"{self._multiple_comparison_correction}"
            )

        tested_results = [
            result
            for result in pairwise_results
            if _supports_p_value_correction(result)
        ]
        if not tested_results:
            return

        eligible_results: list[reports.ComparisonResult] = []
        p_values = []
        for result in tested_results:
            p_value = result.test_result.p_value
            if p_value is None:  # Defensive guard; filtered above.
                continue
            eligible_results.append(result)
            p_values.append(p_value)
        if not p_values:
            return
        corrected_flags = holm_bonferroni(p_values, alpha=self._alpha)
        corrected_p_values = _holm_adjusted_p_values(p_values)

        for result, corrected, corrected_p in zip(
            eligible_results, corrected_flags, corrected_p_values
        ):
            result.corrected_significant = corrected
            result.corrected_p_value = corrected_p
            if corrected:
                if result.delta > 0:
                    result.winner = result.run_a_id
                elif result.delta < 0:
                    result.winner = result.run_b_id
                else:
                    result.winner = "tie"
            else:
                result.winner = "tie"

    def _build_win_loss_matrix(
        self,
        run_ids: Sequence[str],
        metric: str,
        pairwise_results: list[reports.ComparisonResult],
    ) -> reports.WinLossMatrix:
        """Build win/loss matrix for a specific metric.

        Args:
            run_ids: List of run IDs
            metric: Metric name
            pairwise_results: All pairwise comparison results

        Returns:
            WinLossMatrix for the metric
        """
        n = len(run_ids)
        matrix = [["—" for _ in range(n)] for _ in range(n)]

        win_counts = {rid: 0 for rid in run_ids}
        loss_counts = {rid: 0 for rid in run_ids}
        tie_counts = {rid: 0 for rid in run_ids}

        # Fill matrix from pairwise results
        for result in pairwise_results:
            if result.metric_name != metric:
                continue

            idx_a = run_ids.index(result.run_a_id)
            idx_b = run_ids.index(result.run_b_id)

            if result.winner == result.run_a_id:
                matrix[idx_a][idx_b] = "win"
                matrix[idx_b][idx_a] = "loss"
                win_counts[result.run_a_id] += 1
                loss_counts[result.run_b_id] += 1
            elif result.winner == result.run_b_id:
                matrix[idx_a][idx_b] = "loss"
                matrix[idx_b][idx_a] = "win"
                loss_counts[result.run_a_id] += 1
                win_counts[result.run_b_id] += 1
            else:  # tie
                matrix[idx_a][idx_b] = "tie"
                matrix[idx_b][idx_a] = "tie"
                tie_counts[result.run_a_id] += 1
                tie_counts[result.run_b_id] += 1

        return reports.WinLossMatrix(
            run_ids=list(run_ids),
            metric_name=metric,
            matrix=matrix,
            win_counts=win_counts,
            loss_counts=loss_counts,
            tie_counts=tie_counts,
        )


def compare_runs(
    run_ids: Sequence[str],
    *,
    storage_path: str | Path,
    metrics: Sequence[str] | None = None,
    statistical_test: StatisticalTest = StatisticalTest.BOOTSTRAP,
    alpha: float = 0.05,
) -> reports.ComparisonReport:
    """Convenience function to compare runs.

    Args:
        run_ids: List of run IDs to compare
        storage_path: Path to experiment storage
        metrics: List of metrics to compare (None = all)
        statistical_test: Type of statistical test
        alpha: Significance level

    Returns:
        ComparisonReport with all comparisons

    Example:
        >>> report = compare_runs(
        ...     ["run-gpt4", "run-claude"],
        ...     storage_path=".cache/experiments",
        ...     metrics=["ExactMatch", "BLEU"],
        ... )
        >>> print(report.summary())
    """
    engine = ComparisonEngine(
        storage_path=storage_path,
        statistical_test=statistical_test,
        alpha=alpha,
    )

    return engine.compare_runs(run_ids, metrics=metrics)


def _holm_adjusted_p_values(p_values: Sequence[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted((p, idx) for idx, p in enumerate(p_values))
    adjusted_sorted = [0.0] * n
    running_max = 0.0
    for rank, (p_val, _orig_idx) in enumerate(indexed):
        adjusted = (n - rank) * p_val
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted = [0.0] * n
    for rank, (_p_val, orig_idx) in enumerate(indexed):
        adjusted[orig_idx] = adjusted_sorted[rank]
    return adjusted


def _supports_p_value_correction(result: reports.ComparisonResult) -> bool:
    test_result = result.test_result
    if test_result is None:
        return False
    if getattr(test_result, "inference_mode", "hypothesis_test") != "hypothesis_test":
        return False
    return test_result.p_value is not None


__all__ = [
    "ComparisonEngine",
    "compare_runs",
]
