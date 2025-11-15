"""Analysis utilities for prompt engineering experiment results."""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_results_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load experiment results from CSV file."""
    return pd.read_csv(csv_path)


def analyze_prompt_effectiveness(csv_path: Path) -> Dict[str, float]:
    """Analyze prompt effectiveness from CSV results."""
    df = load_results_from_csv(csv_path)

    # Check the actual column names in the CSV
    print("Available columns:", df.columns.tolist())

    # Look for the actual strategy column - based on the CSV output it's 'template_strategy'
    strategy_col = (
        "template_strategy" if "template_strategy" in df.columns else "prompt_strategy"
    )

    # Look for accuracy column - based on the CSV it's 'metric:ExactMatch'
    accuracy_col = (
        "metric:ExactMatch" if "metric:ExactMatch" in df.columns else "exact_match"
    )

    if strategy_col in df.columns and accuracy_col in df.columns:
        results = df.groupby(strategy_col)[accuracy_col].agg(["mean", "count"]).round(3)
        return results.to_dict()
    else:
        print("Required columns not found in CSV")
        print("Available columns:", df.columns.tolist())
        return {}


def compare_models_by_prompt(csv_path: Path) -> pd.DataFrame:
    """Compare model performance by prompt strategy."""
    df = load_results_from_csv(csv_path)

    # Check actual column names
    model_col = "model_identifier" if "model_identifier" in df.columns else "model_name"
    strategy_col = (
        "template_strategy" if "template_strategy" in df.columns else "prompt_strategy"
    )
    accuracy_col = (
        "metric:ExactMatch" if "metric:ExactMatch" in df.columns else "exact_match"
    )

    # Group by both model and prompt strategy to compare effectiveness
    if (
        model_col in df.columns
        and strategy_col in df.columns
        and accuracy_col in df.columns
    ):
        comparison = (
            df.groupby([model_col, strategy_col])[accuracy_col].mean().unstack()
        )
        return comparison
    else:
        print("Required columns not found in CSV")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()


def generate_comparison_report(csv_path: Path, output_path: Path) -> None:
    """Generate a comprehensive comparison report."""
    df = load_results_from_csv(csv_path)

    # Determine actual column names
    strategy_col = (
        "template_strategy" if "template_strategy" in df.columns else "prompt_strategy"
    )
    model_col = "model_identifier" if "model_identifier" in df.columns else "model_name"
    accuracy_col = (
        "metric:ExactMatch" if "metric:ExactMatch" in df.columns else "exact_match"
    )

    with open(output_path, "w") as f:
        f.write("# Prompt Engineering Experiment Analysis Report\n\n")

        # Overall statistics
        f.write("## Overall Results\n")
        if accuracy_col in df.columns:
            overall_accuracy = df[accuracy_col].mean()
            f.write(f"- Overall accuracy: {overall_accuracy:.3f}\n")
            f.write(f"- Total samples: {len(df)}\n\n")

        # By prompt strategy
        f.write("## By Prompt Strategy\n")
        if strategy_col in df.columns and accuracy_col in df.columns:
            by_strategy = df.groupby(strategy_col)[accuracy_col].agg(["mean", "count"])
            for strategy, row in by_strategy.iterrows():
                f.write(f"- {strategy}: {row['mean']:.3f} (n={int(row['count'])})\n")
            f.write("\n")

        # By model
        f.write("## By Model\n")
        if model_col in df.columns and accuracy_col in df.columns:
            by_model = df.groupby(model_col)[accuracy_col].agg(["mean", "count"])
            for model, row in by_model.iterrows():
                f.write(f"- {model}: {row['mean']:.3f} (n={int(row['count'])})\n")
            f.write("\n")

        # Cross-analysis
        f.write("## Cross Analysis (Model vs Strategy)\n")
        if (
            model_col in df.columns
            and strategy_col in df.columns
            and accuracy_col in df.columns
        ):
            cross_analysis = (
                df.groupby([model_col, strategy_col])[accuracy_col].mean().unstack()
            )
            try:
                if hasattr(cross_analysis, "to_markdown"):
                    f.write(cross_analysis.to_markdown())
                else:
                    f.write(str(cross_analysis))
            except ImportError:
                # If tabulate is not available, just write the string representation
                f.write(str(cross_analysis))
            f.write("\n\n")

        # List available columns for reference
        f.write("## Available Columns in Dataset\n")
        for col in df.columns:
            f.write(f"- {col}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python results_analysis.py <input_csv> <output_report>")
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_report = Path(sys.argv[2])

    generate_comparison_report(input_csv, output_report)
    print(f"Analysis report saved to {output_report}")
