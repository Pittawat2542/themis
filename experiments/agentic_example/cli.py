"""CLI for the agentic experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Sequence

from cyclopts import App, Parameter

from themis.experiment import export as experiment_export

from .config import AGENTIC_DEFAULT_CONFIG, AgenticExperimentConfig, load_config
from .experiment import run_experiment

app = App(help="Agentic experiment CLI")


@app.command()
def run(
    *,
    config_path: Annotated[Path | None, Parameter(help="Path to config file")] = None,
    run_id: Annotated[str | None, Parameter(help="Override run identifier")] = None,
    storage_dir: Annotated[
        Path | None, Parameter(help="Override storage directory")
    ] = None,
    resume: Annotated[bool | None, Parameter(help="Toggle resume")] = None,
    planner_prompt: Annotated[
        str | None, Parameter(help="Override planner prompt")
    ] = None,
    final_prefix: Annotated[
        str | None, Parameter(help="Override final prompt prefix")
    ] = None,
    dry_run: Annotated[bool, Parameter(help="Show plan without running")] = False,
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
        planner_prompt=planner_prompt,
        final_prompt_prefix=final_prefix,
    )

    if dry_run:
        _print_plan_summary(config)
        return 0

    report = run_experiment(config)
    exact = report.evaluation_report.metrics.get("ExactMatch")
    length_metric = report.evaluation_report.metrics.get("ResponseLength")
    if exact:
        print(f"Exact match mean: {exact.mean:.3f}")
    if length_metric:
        avg_length = length_metric.mean if length_metric.count else 0.0
        print(f"Average response length: {avg_length:.1f}")
    _export_report_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="Agentic experiment",
    )
    return 0


def _load_config(path: Path | None) -> AgenticExperimentConfig:
    if path is None:
        return AGENTIC_DEFAULT_CONFIG
    return load_config(path)


def _print_plan_summary(config: AgenticExperimentConfig) -> None:
    model_names = ", ".join(model.name for model in config.models)
    print("Models:", model_names)
    print("Datasets:", ", ".join(dataset.name for dataset in config.datasets))
    print("Planner prompt:", config.planner_prompt)
    print("Final prompt prefix:", config.final_prompt_prefix)
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
    except SystemExit as exc:  # pragma: no cover
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
