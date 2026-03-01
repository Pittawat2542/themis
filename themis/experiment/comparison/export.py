"""Export mixin for multi-experiment comparison."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from themis.experiment.comparison.entities import ComparisonRow


class ComparisonExportMixin:
    """Mixin for exporting comparison data."""

    # These fields are expected to be provided by the concrete class
    experiments: list["ComparisonRow"]
    metrics: list[str]

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
