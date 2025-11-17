"""Multiple-choice question benchmark commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable, Literal, Sequence

from cyclopts import Parameter

from themis.cli.utils import effective_total, export_outputs
from themis.datasets import (
    mmlu_pro as mmlu_pro_dataset,
)
from themis.datasets import (
    super_gpqa as super_gpqa_dataset,
)
from themis.experiment import mcq as mcq_experiment
from themis.experiment import storage as experiment_storage
from themis.utils.logging_utils import configure_logging
from themis.utils.progress import ProgressReporter


def load_multiple_choice_dataset(
    *,
    loader: Callable[..., Sequence],
    source: Literal["huggingface", "local"],
    data_dir: Path | None,
    split: str,
    limit: int | None,
    subjects: Sequence[str] | None,
):
    """Load multiple choice dataset.

    Args:
        loader: Dataset loader function
        source: Dataset source
        data_dir: Directory containing local dataset
        split: Dataset split
        limit: Max rows to load
        subjects: Subjects to filter

    Returns:
        List of generation examples
    """
    if source == "local" and data_dir is None:
        raise ValueError(
            "The --data-dir option is required when --source=local so Themis "
            "knows where to read the dataset."
        )
    samples = loader(
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subjects,
    )
    return [sample.to_generation_example() for sample in samples]


def supergpqa_command(
    *,
    source: Annotated[
        Literal["huggingface", "local"], Parameter(help="Dataset source")
    ] = "huggingface",
    split: Annotated[str, Parameter(help="Dataset split to load")] = "test",
    data_dir: Annotated[
        Path | None, Parameter(help="Directory containing local dataset")
    ] = None,
    limit: Annotated[int | None, Parameter(help="Max rows to load")] = None,
    subjects: Annotated[
        tuple[str, ...], Parameter(help="Subjects or categories to filter")
    ] = (),
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
    """Run the SuperGPQA multiple-choice evaluation."""
    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = load_multiple_choice_dataset(
        loader=super_gpqa_dataset.load_super_gpqa,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    storage_impl = experiment_storage.ExperimentStorage(storage) if storage else None
    experiment = mcq_experiment.build_multiple_choice_json_experiment(
        dataset_name="supergpqa",
        task_id="supergpqa",
        temperature=temperature,
        storage=storage_impl,
    )

    total = effective_total(len(rows), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            rows,
            max_samples=max_samples,
            run_id=run_id,
            resume=resume,
            on_result=progress.on_result,
        )
    print(mcq_experiment.summarize_report(report))
    export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="supergpqa experiment",
    )
    return 0


def mmlu_pro_command(
    *,
    source: Annotated[
        Literal["huggingface", "local"], Parameter(help="Dataset source")
    ] = "huggingface",
    split: Annotated[str, Parameter(help="Dataset split to load")] = "test",
    data_dir: Annotated[
        Path | None, Parameter(help="Directory containing local dataset")
    ] = None,
    limit: Annotated[int | None, Parameter(help="Max rows to load")] = None,
    subjects: Annotated[
        tuple[str, ...], Parameter(help="Subjects or categories to filter")
    ] = (),
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
    """Run the MMLU-Pro multiple-choice evaluation."""
    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = load_multiple_choice_dataset(
        loader=mmlu_pro_dataset.load_mmlu_pro,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    storage_impl = experiment_storage.ExperimentStorage(storage) if storage else None
    experiment = mcq_experiment.build_multiple_choice_json_experiment(
        dataset_name="mmlu-pro",
        task_id="mmlu_pro",
        temperature=temperature,
        storage=storage_impl,
    )

    total = effective_total(len(rows), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            rows,
            max_samples=max_samples,
            run_id=run_id,
            resume=resume,
            on_result=progress.on_result,
        )
    print(mcq_experiment.summarize_report(report))
    export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="mmlu_pro experiment",
    )
    return 0
