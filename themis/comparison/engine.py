"""Comparison engine for analyzing multiple experiment runs.

This module provides the main ComparisonEngine class that orchestrates
loading runs, computing statistics, and generating comparison reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from themis.comparison import reports, statistics
from themis.comparison.statistics import StatisticalTest
from themis.experiment import storage as experiment_storage


class ComparisonEngine:
    """Engine for comparing multiple experiment runs.
    
    This class loads experiment results from storage and performs
    pairwise comparisons across all metrics with statistical testing.
    """
    
    def __init__(
        self,
        *,
        storage: experiment_storage.ExperimentStorage | None = None,
        storage_path: str | Path | None = None,
        statistical_test: StatisticalTest = StatisticalTest.BOOTSTRAP,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000,
    ):
        """Initialize comparison engine.
        
        Args:
            storage: Experiment storage instance
            storage_path: Path to storage (if storage not provided)
            statistical_test: Type of statistical test to use
            alpha: Significance level for tests
            n_bootstrap: Number of bootstrap iterations
            n_permutations: Number of permutations for permutation test
        """
        if storage is None and storage_path is None:
            raise ValueError("Either storage or storage_path must be provided")
        
        self._storage = storage or experiment_storage.ExperimentStorage(storage_path)
        self._statistical_test = statistical_test
        self._alpha = alpha
        self._n_bootstrap = n_bootstrap
        self._n_permutations = n_permutations
    
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
            raise ValueError("Need at least 2 runs to compare")
        
        # Load all runs
        run_data = {}
        for run_id in run_ids:
            try:
                data = self._load_run_metrics(run_id)
                run_data[run_id] = data
            except FileNotFoundError:
                raise ValueError(f"Run not found: {run_id}")
        
        # Determine metrics to compare
        if metrics is None:
            # Use all metrics that appear in all runs
            all_metrics = set(run_data[run_ids[0]].keys())
            for run_id in run_ids[1:]:
                all_metrics &= set(run_data[run_id].keys())
            metrics = sorted(all_metrics)
        
        if not metrics:
            raise ValueError("No common metrics found across all runs")
        
        # Perform pairwise comparisons
        pairwise_results = []
        for metric in metrics:
            for i, run_a in enumerate(run_ids):
                for run_b in run_ids[i + 1:]:
                    result = self._compare_pair(
                        run_a,
                        run_b,
                        metric,
                        run_data[run_a][metric],
                        run_data[run_b][metric],
                        statistical_test or self._statistical_test,
                    )
                    pairwise_results.append(result)
        
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
                key=lambda rid: sum(run_data[rid][metric]) / len(run_data[rid][metric])
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
                "statistical_test": self._statistical_test.value,
                "alpha": self._alpha,
                "n_runs": len(run_ids),
                "n_metrics": len(metrics),
            },
        )
    
    def _load_run_metrics(self, run_id: str) -> dict[str, list[float]]:
        """Load all metric scores for a run.
        
        Returns:
            Dictionary mapping metric names to lists of scores
        """
        # Load evaluation records from storage (returns dict of cache_key -> EvaluationRecord)
        eval_dict = self._storage.load_cached_evaluations(run_id)
        
        # Organize scores by metric
        metric_scores: dict[str, list[float]] = {}
        
        # eval_dict is a dict, so iterate over values
        for record in eval_dict.values():
            for metric_name, score_obj in record.scores.items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                
                # Get numeric score
                if hasattr(score_obj, 'value'):
                    score = score_obj.value
                elif isinstance(score_obj, (int, float)):
                    score = float(score_obj)
                else:
                    continue  # Skip non-numeric scores
                
                metric_scores[metric_name].append(score)
        
        return metric_scores
    
    def _compare_pair(
        self,
        run_a_id: str,
        run_b_id: str,
        metric_name: str,
        samples_a: list[float],
        samples_b: list[float],
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
        # Calculate means
        mean_a = sum(samples_a) / len(samples_a)
        mean_b = sum(samples_b) / len(samples_b)
        
        # Calculate delta
        delta = mean_a - mean_b
        delta_percent = (delta / mean_b * 100) if mean_b != 0 else 0.0
        
        # Perform statistical test
        test_result = None
        if test_type == StatisticalTest.T_TEST:
            test_result = statistics.t_test(
                samples_a, samples_b, alpha=self._alpha, paired=True
            )
        elif test_type == StatisticalTest.BOOTSTRAP:
            test_result = statistics.bootstrap_confidence_interval(
                samples_a,
                samples_b,
                n_bootstrap=self._n_bootstrap,
                confidence_level=1 - self._alpha,
            )
        elif test_type == StatisticalTest.PERMUTATION:
            test_result = statistics.permutation_test(
                samples_a,
                samples_b,
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
            run_a_samples=samples_a,
            run_b_samples=samples_b,
        )
    
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


__all__ = [
    "ComparisonEngine",
    "compare_runs",
]
