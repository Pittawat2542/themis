"""Leaderboard generation for benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.experiment.comparison import compare_experiments


def leaderboard_command(
    *,
    run_ids: Annotated[list[str], Parameter(help="Run IDs to include in leaderboard")],
    storage: Annotated[Path, Parameter(help="Storage directory")] = Path(".cache/runs"),
    metric: Annotated[str, Parameter(help="Primary metric for ranking")] = "accuracy",
    format: Annotated[
        str, Parameter(help="Output format: markdown, latex, csv")
    ] = "markdown",
    output: Annotated[
        Path | None, Parameter(help="Output file path (optional)")
    ] = None,
    title: Annotated[str, Parameter(help="Leaderboard title")] = "Leaderboard",
    ascending: Annotated[
        bool, Parameter(help="Rank in ascending order (lower is better)")
    ] = False,
    include_cost: Annotated[bool, Parameter(help="Include cost column")] = True,
    include_metadata: Annotated[
        list[str] | None,
        Parameter(help="Metadata fields to include (e.g., model, temperature)"),
    ] = None,
) -> int:
    """Generate benchmark leaderboard from experiment runs.

    Creates a ranked table of experiments based on a primary metric.
    Perfect for README files, documentation, and benchmark tracking.

    Examples:
        # Basic leaderboard
        uv run python -m themis.cli leaderboard \\
          --run-ids run-1 run-2 run-3 \\
          --metric accuracy \\
          --output LEADERBOARD.md

        # With custom metadata
        uv run python -m themis.cli leaderboard \\
          --run-ids run-gpt4 run-claude run-gemini \\
          --metric accuracy \\
          --include-metadata model \\
          --include-metadata temperature \\
          --output results.md

        # LaTeX for papers
        uv run python -m themis.cli leaderboard \\
          --run-ids run-1 run-2 run-3 \\
          --metric accuracy \\
          --format latex \\
          --title "Math500 Benchmark Results" \\
          --output leaderboard.tex

        # Cost-optimized ranking (lower is better)
        uv run python -m themis.cli leaderboard \\
          --run-ids run-1 run-2 run-3 \\
          --metric cost \\
          --ascending true \\
          --output cost_leaderboard.md
    """
    try:
        # Load experiments
        print(f"Loading experiments from {storage}...")
        comparison = compare_experiments(
            run_ids=run_ids,
            storage_dir=storage,
            include_metadata=True,
        )

        print(f"✓ Loaded {len(comparison.experiments)} experiments")

        # Rank by metric
        ranked = comparison.rank_by_metric(metric, ascending=ascending)

        print(f"✓ Ranked by {metric} ({'ascending' if ascending else 'descending'})")

        # Generate leaderboard
        if format == "markdown":
            content = _generate_markdown_leaderboard(
                ranked=ranked,
                metric=metric,
                title=title,
                include_cost=include_cost,
                include_metadata=include_metadata,
                comparison=comparison,
            )
        elif format == "latex":
            content = _generate_latex_leaderboard(
                ranked=ranked,
                metric=metric,
                title=title,
                include_cost=include_cost,
                include_metadata=include_metadata,
                comparison=comparison,
            )
        elif format == "csv":
            content = _generate_csv_leaderboard(
                ranked=ranked,
                metric=metric,
                include_cost=include_cost,
                include_metadata=include_metadata,
                comparison=comparison,
            )
        else:
            print(f"Error: Unknown format '{format}'")
            print("Available formats: markdown, latex, csv")
            return 1

        # Output
        if output:
            output = Path(output)
            output.write_text(content, encoding="utf-8")
            print(f"\n✓ Leaderboard saved to {output}")
        else:
            print("\n" + "=" * 80)
            print(content)
            print("=" * 80)

        # Show top 3
        print(f"\n🏆 Top 3 by {metric}:")
        for i, exp in enumerate(ranked[:3], 1):
            value = exp.get_metric(metric)
            emoji = ["🥇", "🥈", "🥉"][i - 1]
            print(f"  {emoji} {i}. {exp.run_id}: {value:.4f}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def _generate_markdown_leaderboard(
    ranked,
    metric: str,
    title: str,
    include_cost: bool,
    include_metadata: list[str] | None,
    comparison,
) -> str:
    """Generate markdown leaderboard."""
    lines = [f"# {title}\n"]

    # Date
    from datetime import datetime

    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Build header
    headers = ["Rank", "Run ID", metric.capitalize()]

    # Add metadata columns
    if include_metadata:
        headers.extend(include_metadata)

    # Add cost if requested
    has_cost = include_cost and any(exp.get_cost() is not None for exp in ranked)
    if has_cost:
        headers.append("Cost ($)")

    # Add other metrics
    other_metrics = [m for m in comparison.metrics if m != metric]
    headers.extend(other_metrics)

    headers.extend(["Samples", "Failures"])

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Build rows
    for rank, exp in enumerate(ranked, 1):
        values = [str(rank), exp.run_id]

        # Primary metric
        val = exp.get_metric(metric)
        values.append(f"**{val:.4f}**" if val is not None else "N/A")

        # Metadata
        if include_metadata:
            for field in include_metadata:
                val = exp.metadata.get(field, "—")
                values.append(str(val))

        # Cost
        if has_cost:
            cost = exp.get_cost()
            values.append(f"{cost:.4f}" if cost is not None else "—")

        # Other metrics
        for m in other_metrics:
            val = exp.get_metric(m)
            values.append(f"{val:.4f}" if val is not None else "N/A")

        values.append(str(exp.sample_count))
        values.append(str(exp.failure_count))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _generate_latex_leaderboard(
    ranked,
    metric: str,
    title: str,
    include_cost: bool,
    include_metadata: list[str] | None,
    comparison,
) -> str:
    """Generate LaTeX leaderboard."""
    lines = []

    # Calculate columns
    n_cols = 2  # Rank + Run ID
    n_cols += 1  # Primary metric

    if include_metadata:
        n_cols += len(include_metadata)

    has_cost = include_cost and any(exp.get_cost() is not None for exp in ranked)
    if has_cost:
        n_cols += 1

    other_metrics = [m for m in comparison.metrics if m != metric]
    n_cols += len(other_metrics)
    n_cols += 2  # Samples + Failures

    # Table preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{title}}}")
    lines.append("\\label{tab:leaderboard}")

    col_spec = "c" + "l" + "r" * (n_cols - 2)  # Center rank, left run_id, right numbers
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    headers = ["Rank", "Run ID", f"\\textbf{{{metric}}}"]

    if include_metadata:
        headers.extend(include_metadata)

    if has_cost:
        headers.append("Cost (\\$)")

    headers.extend(other_metrics)
    headers.extend(["Samples", "Failures"])

    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    # Rows
    for rank, exp in enumerate(ranked, 1):
        values = [str(rank), exp.run_id.replace("_", "\\_")]

        # Primary metric (bold)
        val = exp.get_metric(metric)
        values.append(f"\\textbf{{{val:.4f}}}" if val is not None else "---")

        # Metadata
        if include_metadata:
            for field in include_metadata:
                val = exp.metadata.get(field, "---")
                val_str = str(val).replace("_", "\\_")
                values.append(val_str)

        # Cost
        if has_cost:
            cost = exp.get_cost()
            values.append(f"{cost:.4f}" if cost is not None else "---")

        # Other metrics
        for m in other_metrics:
            val = exp.get_metric(m)
            values.append(f"{val:.4f}" if val is not None else "---")

        values.append(str(exp.sample_count))
        values.append(str(exp.failure_count))

        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _generate_csv_leaderboard(
    ranked,
    metric: str,
    include_cost: bool,
    include_metadata: list[str] | None,
    comparison,
) -> str:
    """Generate CSV leaderboard."""
    import csv
    import io

    output = io.StringIO()

    # Build header
    headers = ["rank", "run_id", metric]

    if include_metadata:
        headers.extend(include_metadata)

    has_cost = include_cost and any(exp.get_cost() is not None for exp in ranked)
    if has_cost:
        headers.append("cost")

    other_metrics = [m for m in comparison.metrics if m != metric]
    headers.extend(other_metrics)
    headers.extend(["sample_count", "failure_count"])

    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    # Write rows
    for rank, exp in enumerate(ranked, 1):
        row = {"rank": rank, "run_id": exp.run_id}

        # Primary metric
        val = exp.get_metric(metric)
        row[metric] = val if val is not None else ""

        # Metadata
        if include_metadata:
            for field in include_metadata:
                row[field] = exp.metadata.get(field, "")

        # Cost
        if has_cost:
            cost = exp.get_cost()
            row["cost"] = cost if cost is not None else ""

        # Other metrics
        for m in other_metrics:
            val = exp.get_metric(m)
            row[m] = val if val is not None else ""

        row["sample_count"] = exp.sample_count
        row["failure_count"] = exp.failure_count

        writer.writerow(row)

    return output.getvalue()


__all__ = ["leaderboard_command"]
