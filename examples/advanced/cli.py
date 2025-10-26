"""CLI for the advanced experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Sequence

from cyclopts import App, Parameter

from themis.experiment import export as experiment_export

from .config import ADVANCED_DEFAULT_CONFIG, AdvancedExperimentConfig, load_config
from .experiment import run_experiment, summarize_subject_breakdown

app = App(help="Advanced Themis experiment CLI")


@app.command()
def run(
    *,
    config_path: Annotated[Path | None, Parameter(help="Path to JSON config")] = None,
    prompt_style: Annotated[str | None, Parameter(help="Override prompt style")] = None,
    disable_subject_breakdown: Annotated[
        bool, Parameter(help="Disable breakdown")
    ] = False,
    run_id: Annotated[str | None, Parameter(help="Override run ID")] = None,
    storage_dir: Annotated[
        Path | None, Parameter(help="Override storage directory")
    ] = None,
    resume: Annotated[bool | None, Parameter(help="Override resume toggle")] = None,
    attempts: Annotated[
        int | None, Parameter(help="Override test-time attempts")
    ] = None,
    dry_run: Annotated[
        bool, Parameter(help="Preview configuration without running")
    ] = False,
    csv_output: Annotated[
        Path | None, Parameter(help="Write CSV export to this path")
    ] = None,
    html_output: Annotated[
        Path | None, Parameter(help="Write HTML summary to this path")
    ] = None,
    json_output: Annotated[
        Path | None, Parameter(help="Write JSON export to this path")
    ] = None,
) -> int:
    config = _load_config(config_path)
    config = config.apply_overrides(
        run_id=run_id,
        storage_dir=storage_dir,
        resume=resume,
        enable_subject_breakdown=False if disable_subject_breakdown else None,
        prompt_style=prompt_style,
        test_time_attempts=attempts,
    )

    if dry_run:
        _print_plan_summary(config)
        return 0

    report = run_experiment(config)
    print(summarize_subject_breakdown(report))
    if config.enable_subject_breakdown:
        breakdown = report.metadata.get("subject_breakdown", {})
        for subject, score in breakdown.items():
            print(f"  - {subject}: {score:.3f}")
    _export_report_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="Advanced experiment",
    )
    return 0


def _load_config(path: Path | None) -> AdvancedExperimentConfig:
    if path is None:
        return ADVANCED_DEFAULT_CONFIG
    return load_config(path)


def _print_plan_summary(config: AdvancedExperimentConfig) -> None:
    model_names = ", ".join(model.name for model in config.models)
    dataset_names = ", ".join(dataset.name for dataset in config.datasets)
    print("Configured models:", model_names)
    print("Datasets:", dataset_names)
    print("Prompt style:", config.prompt_style)
    print("Test-time attempts:", config.test_time_attempts)
    print("Storage:", config.storage_dir)


def _export_report_outputs(
    report,
    *,
    csv_output: Path | None,
    html_output: Path | None,
    json_output: Path | None,
    title: str,
) -> None:
    outputs = experiment_export.export_report_bundle(
        report,
        csv_path=csv_output,
        html_path=html_output,
        json_path=json_output,
        title=title,
    )
    for kind, output_path in outputs.items():
        print(f"Exported {kind.upper()} to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parsed = list(argv) if argv is not None else None
    try:
        result = app(parsed)
    except SystemExit as exc:  # pragma: no cover - CLI integration
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
