"""CLI wrapper for the prompt engineering experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Sequence

from cyclopts import App, Parameter

from themis.experiment import export as experiment_export
from themis.generation.providers import litellm_provider  # noqa: F401

from .config import DEFAULT_CONFIG, PromptEngineeringConfig, load_config
from .experiment import run_experiment, summarize_report, analyze_by_prompt_strategy

app = App(help="Run the prompt engineering experiment")


@app.command()
def run(
    *,
    config_path: Annotated[
        Path | None, Parameter(help="Path to JSON/YAML configuration file")
    ] = None,
    run_id: Annotated[str | None, Parameter(help="Override run identifier")] = None,
    storage_dir: Annotated[
        Path | None, Parameter(help="Override storage directory")
    ] = None,
    resume: Annotated[
        bool | None, Parameter(help="Toggle resumability/caching")
    ] = None,
    dry_run: Annotated[bool, Parameter(help="Print plan without running")] = False,
    csv_output: Annotated[
        Path | None, Parameter(help="Write per-sample CSV export to this path")
    ] = None,
    html_output: Annotated[
        Path | None, Parameter(help="Write HTML summary to this path")
    ] = None,
    json_output: Annotated[
        Path | None, Parameter(help="Write JSON export to this path")
    ] = None,
    analyze: Annotated[bool, Parameter(help="Show prompt strategy analysis")] = False,
) -> int:
    """Run the prompt engineering experiment."""
    
    config = _load_config(config_path)
    config = config.apply_overrides(
        run_id=run_id, storage_dir=storage_dir, resume=resume
    )

    if dry_run:
        _print_plan_summary(config)
        return 0

    print(f"Running prompt engineering experiment: {config.run_id}")
    print(f"Storage: {config.storage_dir}")
    print(f"Resume: {config.resume}")
    print()
    
    report = run_experiment(config)
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(summarize_report(report))
    
    if analyze:
        print()
        analyze_by_prompt_strategy(report)
    
    _export_report_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="Prompt Engineering Experiment",
    )
    return 0


def _load_config(path: Path | None) -> PromptEngineeringConfig:
    """Load configuration from file or use default."""
    if path is None:
        return DEFAULT_CONFIG
    return load_config(path)


def _print_plan_summary(config: PromptEngineeringConfig) -> None:
    """Print a summary of the experiment plan."""
    print("PROMPT ENGINEERING EXPERIMENT PLAN")
    print("="*50)
    
    print(f"Run ID: {config.run_id}")
    print(f"Storage: {config.storage_dir}")
    print(f"Resume: {config.resume}")
    
    print("\nPrompt Variants:")
    for variant in config.prompt_variants:
        print(f"  - {variant.name}: {variant.description}")
    
    print("\nModels:")
    for model in config.models:
        print(f"  - {model.name} ({model.provider}): {model.description}")
    
    print(f"\nSampling Strategies: {len(config.samplings)}")
    for sampling in config.samplings:
        print(f"  - {sampling.name}: temp={sampling.temperature}, max_tokens={sampling.max_tokens}")
    
    print(f"\nDatasets: {len(config.datasets)}")
    for dataset in config.datasets:
        print(f"  - {dataset.name} ({dataset.kind}): limit={dataset.limit or 'all'}")


def _export_report_outputs(
    report,
    *,
    csv_output: Path | None,
    html_output: Path | None,
    json_output: Path | None,
    title: str,
) -> None:
    """Export report outputs to specified formats."""
    outputs = experiment_export.export_report_bundle(
        report,
        csv_path=csv_output,
        html_path=html_output,
        json_path=json_output,
        title=title,
    )
    if outputs:
        print("\nEXPORTED RESULTS:")
        for kind, output_path in outputs.items():
            print(f"  {kind.upper()}: {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parsed = list(argv) if argv is not None else None
    try:
        result = app(parsed)
    except SystemExit as exc:  # pragma: no cover - CLI integration path
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())