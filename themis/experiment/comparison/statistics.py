"""Statistical comparison and loaders for multi-experiment data."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from themis.core.entities import ExperimentReport
from themis.exceptions import ConfigurationError, MetricError
from themis.experiment.comparison.entities import ComparisonRow, ConfigDiff
from themis.experiment.comparison.export import ComparisonExportMixin


@dataclass
class MultiExperimentComparison(ComparisonExportMixin):
    """Comparison across multiple experiments."""

    experiments: list[ComparisonRow]
    metrics: list[str]

    def __post_init__(self):
        """Validate comparison data."""
        if not self.experiments:
            raise ConfigurationError("Must have at least one experiment to compare")
        if not self.metrics:
            # Infer metrics from first experiment
            if self.experiments:
                self.metrics = list(self.experiments[0].metric_values.keys())

    def rank_by_metric(
        self, metric: str, ascending: bool = False
    ) -> list[ComparisonRow]:
        """Rank experiments by metric value.

        Args:
            metric: Metric name to rank by (can be 'cost' or 'total_cost'
                for cost ranking)
            ascending: If True, rank from lowest to highest (default: False)

        Returns:
            List of experiments sorted by metric value
        """
        # Special handling for cost metrics
        if metric not in self.metrics and metric not in ("cost", "total_cost"):
            raise MetricError(f"Metric '{metric}' not found. Available: {self.metrics}")

        # Sort experiments, handling None values
        def key_func(row: ComparisonRow) -> tuple[bool, float]:
            value = row.get_metric(metric)
            # Put None values at the end
            if value is None:
                return (True, float("inf"))
            return (False, value)

        return sorted(self.experiments, key=key_func, reverse=not ascending)

    def highlight_best(
        self, metric: str, higher_is_better: bool = True
    ) -> ComparisonRow | None:
        """Find experiment with best value for metric.

        Args:
            metric: Metric name
            higher_is_better: If True, higher values are better (default: True)

        Returns:
            Experiment with best metric value, or None if no valid values
        """
        ranked = self.rank_by_metric(metric, ascending=not higher_is_better)
        # Return first experiment with valid metric value
        for exp in ranked:
            if exp.get_metric(metric) is not None:
                return exp
        return None

    def pareto_frontier(
        self, objectives: list[str], maximize: list[bool] | None = None
    ) -> list[str]:
        """Find Pareto-optimal experiments.

        Args:
            objectives: List of metric names to optimize
            maximize: For each objective, whether to maximize (True) or
                minimize (False). Default: maximize all objectives.

        Returns:
            List of run_ids on the Pareto frontier
        """
        if not objectives:
            raise ConfigurationError("Must specify at least one objective")

        if maximize is None:
            maximize = [True] * len(objectives)

        if len(maximize) != len(objectives):
            raise ConfigurationError(
                f"maximize list length ({len(maximize)}) must match "
                f"objectives length ({len(objectives)})"
            )

        # Filter out experiments with missing values
        valid_experiments = [
            exp
            for exp in self.experiments
            if all(exp.get_metric(obj) is not None for obj in objectives)
        ]

        if not valid_experiments:
            return []

        pareto_optimal: list[ComparisonRow] = []

        for candidate in valid_experiments:
            is_dominated = False

            # Check if candidate is dominated by any other experiment
            for other in valid_experiments:
                if candidate.run_id == other.run_id:
                    continue

                # Check if 'other' dominates 'candidate'
                dominates = True
                strictly_better_in_one = False

                for obj, should_maximize in zip(objectives, maximize, strict=True):
                    candidate_val = candidate.get_metric(obj)
                    other_val = other.get_metric(obj)

                    # Should never be None due to filtering, but handle defensively
                    if candidate_val is None or other_val is None:
                        dominates = False
                        break

                    if should_maximize:
                        if other_val < candidate_val:
                            dominates = False
                            break
                        if other_val > candidate_val:
                            strictly_better_in_one = True
                    else:
                        if other_val > candidate_val:
                            dominates = False
                            break
                        if other_val < candidate_val:
                            strictly_better_in_one = True

                if dominates and strictly_better_in_one:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(candidate)

        return [exp.run_id for exp in pareto_optimal]


def load_experiment_report(storage_dir: Path, run_id: str) -> ExperimentReport | None:
    """Load experiment report from storage.

    Args:
        storage_dir: Storage directory
        run_id: Run identifier

    Returns:
        ExperimentReport if found, None otherwise
    """
    report_path = storage_dir / run_id / "report.json"

    if not report_path.exists():
        return None

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct ExperimentReport from JSON
    # Note: This is a simplified loader. For production,
    # you'd want proper deserialization
    return data


def compare_experiments(
    run_ids: list[str],
    storage_dir: Path | str,
    metrics: list[str] | None = None,
    include_metadata: bool = True,
) -> MultiExperimentComparison:
    """Compare multiple experiments.

    Args:
        run_ids: List of experiment run IDs to compare
        storage_dir: Directory containing experiment results
        metrics: Metrics to compare (None = all available)
        include_metadata: Include config metadata in comparison

    Returns:
        Comparison object with all experiment data

    Raises:
        FileNotFoundError: If experiment data not found
        ValueError: If no valid experiments found
    """
    storage_dir = Path(storage_dir)

    comparison_rows: list[ComparisonRow] = []
    all_metrics: set[str] = set()

    for run_id in run_ids:
        # Load evaluation records
        try:
            # Try loading from a report.json if available
            report_path = storage_dir / run_id / "report.json"
            if report_path.exists():
                with report_path.open("r", encoding="utf-8") as f:
                    report_data = json.load(f)

                metric_values: dict[str, float] = {}
                # The JSON structure has a "metrics" array with {name, count, mean}
                if "metrics" in report_data:
                    for metric_data in report_data["metrics"]:
                        if isinstance(metric_data, dict):
                            metric_name = metric_data.get("name")
                            metric_mean = metric_data.get("mean")
                            if metric_name and metric_mean is not None:
                                metric_values[metric_name] = metric_mean
                                all_metrics.add(metric_name)

                metadata_dict: dict[str, Any] = {}
                if include_metadata and "summary" in report_data:
                    # The summary section contains metadata
                    metadata_dict = report_data.get("summary", {})

                # Count samples and failures
                sample_count = report_data.get("total_samples", 0)
                failure_count = report_data.get("summary", {}).get(
                    "run_failures", 0
                ) + report_data.get("summary", {}).get("evaluation_failures", 0)

                # Get timestamp from metadata or file modification time
                timestamp = metadata_dict.get("timestamp")
                if not timestamp and report_path.exists():
                    timestamp = datetime.fromtimestamp(
                        report_path.stat().st_mtime
                    ).isoformat()

                row = ComparisonRow(
                    run_id=run_id,
                    metric_values=metric_values,
                    metadata=metadata_dict,
                    timestamp=timestamp,
                    sample_count=sample_count,
                    failure_count=failure_count,
                )
                comparison_rows.append(row)
            else:
                warnings.warn(
                    f"No report.json found for run '{run_id}', skipping",
                    stacklevel=2,
                )

        except Exception as e:
            warnings.warn(f"Failed to load run '{run_id}': {e}", stacklevel=2)
            continue

    if not comparison_rows:
        raise ConfigurationError(
            f"No valid experiments found for run_ids: {run_ids}. "
            "Make sure experiments have been run and saved with report.json files."
        )

    # Filter metrics if specified
    if metrics:
        all_metrics = set(metrics)

    return MultiExperimentComparison(
        experiments=comparison_rows, metrics=sorted(all_metrics)
    )


def diff_configs(run_id_a: str, run_id_b: str, storage_dir: Path | str) -> ConfigDiff:
    """Show configuration differences between two experiments.

    Args:
        run_id_a: First run ID
        run_id_b: Second run ID
        storage_dir: Storage directory

    Returns:
        ConfigDiff object with differences
    """
    storage_dir = Path(storage_dir)

    # Load config files
    config_a_path = storage_dir / run_id_a / "config.json"
    config_b_path = storage_dir / run_id_b / "config.json"

    if not config_a_path.exists():
        raise FileNotFoundError(f"Config not found for run '{run_id_a}'")
    if not config_b_path.exists():
        raise FileNotFoundError(f"Config not found for run '{run_id_b}'")

    with config_a_path.open("r", encoding="utf-8") as f:
        config_a = json.load(f)
    with config_b_path.open("r", encoding="utf-8") as f:
        config_b = json.load(f)

    # Compute differences
    changed: dict[str, tuple[Any, Any]] = {}
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}

    all_keys = set(config_a.keys()) | set(config_b.keys())

    for key in all_keys:
        if key in config_a and key in config_b:
            if config_a[key] != config_b[key]:
                changed[key] = (config_a[key], config_b[key])
        elif key in config_a:
            removed[key] = config_a[key]
        else:
            added[key] = config_b[key]

    return ConfigDiff(
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        changed_fields=changed,
        added_fields=added,
        removed_fields=removed,
    )
