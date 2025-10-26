"""Cyclopts-powered CLI entrypoints for Themis."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Iterable, Literal, Sequence

from cyclopts import App, Parameter

from themis.config import (
    load_dataset_from_config,
    load_experiment_config,
    run_experiment_from_config,
    summarize_report_for_config,
)
from themis.datasets import math500 as math500_dataset
from themis.experiment import export as experiment_export
from themis.experiment import math as math_experiment
from themis.experiment import storage as experiment_storage
from themis.utils.logging_utils import configure_logging
from themis.utils.progress import ProgressReporter

app = App(help="Run Themis experiments from the command line")


@app.command()
def demo(
    *,
    max_samples: Annotated[
        int | None, Parameter(help="Limit number of demo samples")
    ] = None,
    log_level: Annotated[
        str, Parameter(help="Logging level (critical/error/warning/info/debug/trace)")
    ] = "info",
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
    """Run the built-in demo dataset."""

    configure_logging(log_level)
    dataset = [
        {
            "unique_id": "demo-1",
            "problem": "Convert the point (0,3) in rectangular coordinates to polar coordinates.",
            "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
            "subject": "precalculus",
            "level": 2,
        },
        {
            "unique_id": "demo-2",
            "problem": "What is 7 + 5?",
            "answer": "12",
            "subject": "arithmetic",
            "level": 1,
        },
    ]
    experiment = math_experiment.build_math500_zero_shot_experiment()
    total = _effective_total(len(dataset), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            dataset,
            max_samples=max_samples,
            on_result=progress.on_result,
        )
    print(math_experiment.summarize_report(report))
    _export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="Demo experiment",
    )
    return 0


@app.command(name="math500")
def math500_cmd(
    *,
    source: Annotated[
        Literal["huggingface", "local"], Parameter(help="Dataset source")
    ] = "huggingface",
    data_dir: Annotated[
        Path | None, Parameter(help="Directory containing local dataset")
    ] = None,
    limit: Annotated[int | None, Parameter(help="Max rows to load")] = None,
    subjects: Annotated[tuple[str, ...], Parameter(help="Subjects to filter")] = (),
    max_samples: Annotated[int | None, Parameter(help="Maximum samples to run")] = None,
    storage: Annotated[
        Path | None, Parameter(help="Cache directory for datasets/results")
    ] = None,
    run_id: Annotated[str | None, Parameter(help="Identifier for cached run")] = None,
    resume: Annotated[
        bool, Parameter(help="Reuse cached generations when storage is set")
    ] = True,
    temperature: Annotated[float, Parameter(help="Sampling temperature")] = 0.0,
    log_level: Annotated[
        str, Parameter(help="Logging level (critical/error/warning/info/debug/trace)")
    ] = "info",
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
    """Run the zero-shot MATH-500 evaluation."""

    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = _load_math_dataset(
        source=source,
        data_dir=data_dir,
        limit=limit,
        subjects=subject_filter,
    )

    storage_impl = experiment_storage.ExperimentStorage(storage) if storage else None
    experiment = math_experiment.build_math500_zero_shot_experiment(
        temperature=temperature,
        storage=storage_impl,
    )

    total = _effective_total(len(rows), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            rows,
            max_samples=max_samples,
            run_id=run_id,
            resume=resume,
            on_result=progress.on_result,
        )
    print(math_experiment.summarize_report(report))
    _export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="math500 experiment",
    )
    return 0


@app.command(name="run-config")
def run_configured_experiment(
    *,
    config: Annotated[
        Path, Parameter(help="Path to a Hydra/OmegaConf experiment config file")
    ],
    overrides: Annotated[
        tuple[str, ...],
        Parameter(
            help="Optional Hydra-style overrides (e.g. generation.sampling.temperature=0.2)",
            show_default=False,
        ),
    ] = (),
    log_level: Annotated[
        str, Parameter(help="Logging level (critical/error/warning/info/debug/trace)")
    ] = "info",
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
    """Execute an experiment described via config file."""

    configure_logging(log_level)
    experiment_config = load_experiment_config(config, overrides)
    dataset = load_dataset_from_config(experiment_config)
    total = _effective_total(len(dataset), experiment_config.max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = run_experiment_from_config(
            experiment_config,
            dataset=dataset,
            on_result=progress.on_result,
        )
    print(summarize_report_for_config(experiment_config, report))
    _export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title=f"{experiment_config.name} experiment",
    )
    return 0


def _load_math_dataset(
    *,
    source: Literal["huggingface", "local"],
    data_dir: Path | None,
    limit: int | None,
    subjects: Iterable[str] | None,
):
    if source == "local":
        if data_dir is None:
            raise ValueError(
                "The --data-dir option is required when --source=local so Themis "
                "knows where to read the dataset."
            )
        samples = math500_dataset.load_math500(
            source="local",
            data_dir=data_dir,
            limit=limit,
            subjects=subjects,
        )
    else:
        samples = math500_dataset.load_math500(
            source="huggingface",
            limit=limit,
            subjects=subjects,
        )
    return [sample.to_generation_example() for sample in samples]


def _effective_total(total: int, limit: int | None) -> int:
    if limit is None:
        return total
    return min(total, limit)


def _export_outputs(
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
    parsed_argv = list(argv) if argv is not None else None
    try:
        result = app(parsed_argv)
    except SystemExit as exc:  # pragma: no cover - CLI integration path
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
