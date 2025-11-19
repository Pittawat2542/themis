"""Multi-experiment comparison tools for analyzing multiple runs."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from themis.core.entities import ExperimentReport


@dataclass
class ComparisonRow:
    """Single experiment in a multi-experiment comparison."""

    run_id: str
    metric_values: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    sample_count: int = 0
    failure_count: int = 0

    def get_metric(self, metric_name: str) -> float | None:
        """Get metric value by name.

        Special metric names:
        - 'cost' or 'total_cost': Checks metadata first, then metric_values
        - Any other name: Returns from metric_values dict
        """
        # Handle special cost metrics - check metadata first
        if metric_name in ("cost", "total_cost"):
            cost_data = self.metadata.get("cost")
            if cost_data and "total_cost" in cost_data:
                return cost_data["total_cost"]
            # Fall back to metric_values if not in metadata
            # (for backward compatibility and tests)
            if metric_name in self.metric_values:
                return self.metric_values[metric_name]
            return None

        return self.metric_values.get(metric_name)

    def get_cost(self) -> float | None:
        """Get total cost if available.

        Returns:
            Total cost in USD, or None if not tracked
        """
        return self.get_metric("cost")


@dataclass
class ConfigDiff:
    """Differences between two experiment configurations."""

    run_id_a: str
    run_id_b: str
    changed_fields: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    added_fields: dict[str, Any] = field(default_factory=dict)
    removed_fields: dict[str, Any] = field(default_factory=dict)

    def has_differences(self) -> bool:
        """Check if there are any differences."""
        return bool(self.changed_fields or self.added_fields or self.removed_fields)


@dataclass
class MultiExperimentComparison:
    """Comparison across multiple experiments."""

    experiments: list[ComparisonRow]
    metrics: list[str]

    def __post_init__(self):
        """Validate comparison data."""
        if not self.experiments:
            raise ValueError("Must have at least one experiment to compare")
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
            raise ValueError(f"Metric '{metric}' not found. Available: {self.metrics}")

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
            raise ValueError("Must specify at least one objective")

        if maximize is None:
            maximize = [True] * len(objectives)

        if len(maximize) != len(objectives):
            raise ValueError(
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

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "experiments": [
                {
                    "run_id": exp.run_id,
                    "metric_values": exp.metric_values,
                    "metadata": exp.metadata,
                    "timestamp": exp.timestamp,
                    "sample_count": exp.sample_count,
                    "failure_count": exp.failure_count,
                }
                for exp in self.experiments
            ],
            "metrics": self.metrics,
        }

    def to_csv(self, output_path: Path | str, include_metadata: bool = True) -> None:
        """Export comparison to CSV.

        Args:
            output_path: Where to save CSV file
            include_metadata: Whether to include metadata columns
        """
        import csv

        output_path = Path(output_path)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            # Build column names
            columns = ["run_id"] + self.metrics

            if include_metadata:
                # Collect all metadata keys
                all_metadata_keys: set[str] = set()
                for exp in self.experiments:
                    all_metadata_keys.update(exp.metadata.keys())
                metadata_columns = sorted(all_metadata_keys)
                columns.extend(metadata_columns)
                columns.extend(["timestamp", "sample_count", "failure_count"])

            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for exp in self.experiments:
                row: dict[str, Any] = {"run_id": exp.run_id}
                row.update(exp.metric_values)

                if include_metadata:
                    for key in metadata_columns:
                        row[key] = exp.metadata.get(key, "")
                    row["timestamp"] = exp.timestamp or ""
                    row["sample_count"] = exp.sample_count
                    row["failure_count"] = exp.failure_count

                writer.writerow(row)

    def to_markdown(self, output_path: Path | str | None = None) -> str:
        """Export comparison as markdown table.

        Args:
            output_path: Optional path to save markdown file

        Returns:
            Markdown table string
        """
        lines = ["# Experiment Comparison\n"]

        # Check if any experiment has cost data
        has_cost = any(
            exp.metadata.get("cost") and exp.metadata["cost"].get("total_cost")
            for exp in self.experiments
        )

        # Build table header
        headers = ["Run ID"] + self.metrics + ["Samples", "Failures"]
        if has_cost:
            headers.append("Cost ($)")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Build table rows
        for exp in self.experiments:
            values = [exp.run_id]
            for metric in self.metrics:
                val = exp.get_metric(metric)
                values.append(f"{val:.4f}" if val is not None else "N/A")
            values.append(str(exp.sample_count))
            values.append(str(exp.failure_count))

            # Add cost if available
            if has_cost:
                cost = exp.metadata.get("cost", {}).get("total_cost")
                if cost is not None:
                    values.append(f"{cost:.4f}")
                else:
                    values.append("N/A")

            lines.append("| " + " | ".join(values) + " |")

        markdown = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")

        return markdown

    def to_latex(
        self,
        output_path: Path | str | None = None,
        style: str = "booktabs",
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        """Export comparison as LaTeX table.

        Args:
            output_path: Optional path to save LaTeX file
            style: Table style - "booktabs" or "basic"
            caption: Table caption
            label: LaTeX label for referencing

        Returns:
            LaTeX table string

        Example:
            >>> latex = comparison.to_latex(
            ...     caption="Experiment comparison results",
            ...     label="tab:results"
            ... )
        """
        lines = []

        # Check if any experiment has cost data
        has_cost = any(
            exp.metadata.get("cost") and exp.metadata["cost"].get("total_cost")
            for exp in self.experiments
        )

        # Determine number of columns
        n_metrics = len(self.metrics)
        n_cols = 1 + n_metrics + 2  # run_id + metrics + samples + failures
        if has_cost:
            n_cols += 1

        # Table preamble
        if style == "booktabs":
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            if caption:
                lines.append(f"\\caption{{{caption}}}")
            if label:
                lines.append(f"\\label{{{label}}}")

            # Column specification
            col_spec = "l" + "r" * (n_cols - 1)  # Left for run_id, right for numbers
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append("\\toprule")

            # Header
            headers = ["Run ID"] + self.metrics + ["Samples", "Failures"]
            if has_cost:
                headers.append("Cost (\\$)")
            lines.append(" & ".join(headers) + " \\\\")
            lines.append("\\midrule")

            # Data rows
            for exp in self.experiments:
                values = [exp.run_id.replace("_", "\\_")]  # Escape underscores
                for metric in self.metrics:
                    val = exp.get_metric(metric)
                    values.append(f"{val:.4f}" if val is not None else "---")
                values.append(str(exp.sample_count))
                values.append(str(exp.failure_count))

                # Add cost if available
                if has_cost:
                    cost = exp.metadata.get("cost", {}).get("total_cost")
                    if cost is not None:
                        values.append(f"{cost:.4f}")
                    else:
                        values.append("---")

                lines.append(" & ".join(values) + " \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")

        else:  # basic style
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            if caption:
                lines.append(f"\\caption{{{caption}}}")
            if label:
                lines.append(f"\\label{{{label}}}")

            col_spec = "|l|" + "r|" * (n_cols - 1)
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append("\\hline")

            # Header
            headers = ["Run ID"] + self.metrics + ["Samples", "Failures"]
            if has_cost:
                headers.append("Cost (\\$)")
            lines.append(" & ".join(headers) + " \\\\")
            lines.append("\\hline")

            # Data rows
            for exp in self.experiments:
                values = [exp.run_id.replace("_", "\\_")]
                for metric in self.metrics:
                    val = exp.get_metric(metric)
                    values.append(f"{val:.4f}" if val is not None else "---")
                values.append(str(exp.sample_count))
                values.append(str(exp.failure_count))

                if has_cost:
                    cost = exp.metadata.get("cost", {}).get("total_cost")
                    if cost is not None:
                        values.append(f"{cost:.4f}")
                    else:
                        values.append("---")

                lines.append(" & ".join(values) + " \\\\")
                lines.append("\\hline")

            lines.append("\\end{tabular}")
            lines.append("\\end{table}")

        latex = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.write_text(latex, encoding="utf-8")

        return latex


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
        raise ValueError(
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


__all__ = [
    "ComparisonRow",
    "ConfigDiff",
    "MultiExperimentComparison",
    "compare_experiments",
    "diff_configs",
]
