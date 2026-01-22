"""Comparison reports for analyzing experiment results.

This module provides structured reports for comparing multiple runs,
including win/loss matrices, metric deltas, and statistical significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from themis.comparison.statistics import StatisticalTestResult


@dataclass
class ComparisonResult:
    """Result of comparing two runs on a single metric.
    
    Attributes:
        metric_name: Name of the metric being compared
        run_a_id: Identifier for first run
        run_b_id: Identifier for second run
        run_a_mean: Mean value for first run
        run_b_mean: Mean value for second run
        delta: Difference (run_a - run_b)
        delta_percent: Percentage difference
        winner: ID of the winning run ("tie" if no significant difference)
        test_result: Statistical test result (if performed)
        run_a_samples: Individual sample scores for run A
        run_b_samples: Individual sample scores for run B
    """
    
    metric_name: str
    run_a_id: str
    run_b_id: str
    run_a_mean: float
    run_b_mean: float
    delta: float
    delta_percent: float
    winner: str  # run_a_id, run_b_id, or "tie"
    test_result: StatisticalTestResult | None = None
    run_a_samples: list[float] = field(default_factory=list)
    run_b_samples: list[float] = field(default_factory=list)
    
    def is_significant(self) -> bool:
        """Check if the difference is statistically significant."""
        return self.test_result is not None and self.test_result.significant
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        direction = "↑" if self.delta > 0 else "↓" if self.delta < 0 else "="
        
        summary = (
            f"{self.metric_name}: {self.run_a_id} "
            f"({self.run_a_mean:.3f}) vs {self.run_b_id} "
            f"({self.run_b_mean:.3f}) = {direction}{abs(self.delta):.3f} "
            f"({self.delta_percent:+.1f}%)"
        )
        
        if self.test_result:
            sig_marker = "***" if self.is_significant() else "n.s."
            summary += f" [{sig_marker}, p={self.test_result.p_value:.4f}]"
        
        return summary


@dataclass
class WinLossMatrix:
    """Win/loss/tie matrix for comparing multiple runs.
    
    Attributes:
        run_ids: List of run IDs in the matrix
        metric_name: Name of the metric being compared
        matrix: 2D matrix of results
            matrix[i][j] = result of comparing run i vs run j
            Values: "win", "loss", "tie"
        win_counts: Number of wins for each run
        loss_counts: Number of losses for each run
        tie_counts: Number of ties for each run
    """
    
    run_ids: list[str]
    metric_name: str
    matrix: list[list[str]]
    win_counts: dict[str, int] = field(default_factory=dict)
    loss_counts: dict[str, int] = field(default_factory=dict)
    tie_counts: dict[str, int] = field(default_factory=dict)
    
    def get_result(self, run_a: str, run_b: str) -> str:
        """Get comparison result between two runs."""
        try:
            idx_a = self.run_ids.index(run_a)
            idx_b = self.run_ids.index(run_b)
            return self.matrix[idx_a][idx_b]
        except (ValueError, IndexError):
            return "unknown"
    
    def rank_runs(self) -> list[tuple[str, int, int, int]]:
        """Rank runs by wins (descending), then losses (ascending).
        
        Returns:
            List of (run_id, wins, losses, ties) sorted by performance
        """
        rankings = [
            (
                run_id,
                self.win_counts.get(run_id, 0),
                self.loss_counts.get(run_id, 0),
                self.tie_counts.get(run_id, 0),
            )
            for run_id in self.run_ids
        ]
        
        # Sort by wins (desc), then losses (asc)
        rankings.sort(key=lambda x: (-x[1], x[2]))
        return rankings
    
    def to_table(self) -> str:
        """Generate a formatted table representation."""
        lines = []
        
        # Header
        header = f"{'Run':<20} | " + " | ".join(f"{rid:<12}" for rid in self.run_ids)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for i, run_id in enumerate(self.run_ids):
            row = f"{run_id:<20} | "
            row += " | ".join(f"{self.matrix[i][j]:<12}" for j in range(len(self.run_ids)))
            lines.append(row)
        
        # Summary
        lines.append("")
        lines.append("Summary (W/L/T):")
        for run_id, wins, losses, ties in self.rank_runs():
            lines.append(f"  {run_id}: {wins}/{losses}/{ties}")
        
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Comprehensive comparison report for multiple runs.
    
    Attributes:
        run_ids: List of all run IDs being compared
        metrics: List of metric names being compared
        pairwise_results: List of all pairwise comparison results
        win_loss_matrices: Win/loss matrices for each metric
        best_run_per_metric: Best run for each metric
        overall_best_run: Overall best run across all metrics
        metadata: Additional metadata about the comparison
    """
    
    run_ids: list[str]
    metrics: list[str]
    pairwise_results: list[ComparisonResult] = field(default_factory=list)
    win_loss_matrices: dict[str, WinLossMatrix] = field(default_factory=dict)
    best_run_per_metric: dict[str, str] = field(default_factory=dict)
    overall_best_run: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_comparison(
        self, run_a: str, run_b: str, metric: str
    ) -> ComparisonResult | None:
        """Get comparison result for specific runs and metric."""
        for result in self.pairwise_results:
            if (
                result.metric_name == metric
                and result.run_a_id == run_a
                and result.run_b_id == run_b
            ):
                return result
        return None
    
    def get_metric_results(self, metric: str) -> list[ComparisonResult]:
        """Get all comparison results for a specific metric."""
        return [r for r in self.pairwise_results if r.metric_name == metric]
    
    def summary(self, include_details: bool = False) -> str:
        """Generate a human-readable summary of the comparison.
        
        Args:
            include_details: Whether to include detailed pairwise comparisons
        
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall summary
        lines.append(f"Comparing {len(self.run_ids)} runs across {len(self.metrics)} metrics")
        lines.append(f"Runs: {', '.join(self.run_ids)}")
        lines.append(f"Metrics: {', '.join(self.metrics)}")
        lines.append("")
        
        # Best run per metric
        if self.best_run_per_metric:
            lines.append("Best Run Per Metric:")
            for metric, run_id in self.best_run_per_metric.items():
                lines.append(f"  {metric}: {run_id}")
            lines.append("")
        
        # Overall best
        if self.overall_best_run:
            lines.append(f"Overall Best Run: {self.overall_best_run}")
            lines.append("")
        
        # Win/loss matrices
        if self.win_loss_matrices and include_details:
            lines.append("=" * 80)
            lines.append("WIN/LOSS MATRICES")
            lines.append("=" * 80)
            for metric, matrix in self.win_loss_matrices.items():
                lines.append("")
                lines.append(f"Metric: {metric}")
                lines.append("-" * 40)
                lines.append(matrix.to_table())
                lines.append("")
        
        # Pairwise comparisons
        if include_details and self.pairwise_results:
            lines.append("=" * 80)
            lines.append("PAIRWISE COMPARISONS")
            lines.append("=" * 80)
            
            for metric in self.metrics:
                results = self.get_metric_results(metric)
                if results:
                    lines.append("")
                    lines.append(f"Metric: {metric}")
                    lines.append("-" * 40)
                    for result in results:
                        lines.append(f"  {result.summary()}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "run_ids": self.run_ids,
            "metrics": self.metrics,
            "best_run_per_metric": self.best_run_per_metric,
            "overall_best_run": self.overall_best_run,
            "pairwise_results": [
                {
                    "metric": r.metric_name,
                    "run_a": r.run_a_id,
                    "run_b": r.run_b_id,
                    "run_a_mean": r.run_a_mean,
                    "run_b_mean": r.run_b_mean,
                    "delta": r.delta,
                    "delta_percent": r.delta_percent,
                    "winner": r.winner,
                    "significant": r.is_significant(),
                    "p_value": r.test_result.p_value if r.test_result else None,
                }
                for r in self.pairwise_results
            ],
            "win_loss_summary": {
                metric: {
                    "rankings": [
                        {"run_id": rid, "wins": w, "losses": l, "ties": t}
                        for rid, w, l, t in matrix.rank_runs()
                    ]
                }
                for metric, matrix in self.win_loss_matrices.items()
            },
            "metadata": self.metadata,
        }


__all__ = [
    "ComparisonResult",
    "WinLossMatrix",
    "ComparisonReport",
]
