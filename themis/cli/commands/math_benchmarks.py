"""Math benchmark command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable, Literal, Sequence

from cyclopts import Parameter

from themis.cli.utils import effective_total, export_outputs
from themis.datasets import (
    competition_math as competition_math_dataset,
)
from themis.datasets import (
    math500 as math500_dataset,
)
from themis.experiment import math as math_experiment
from themis.experiment import storage as experiment_storage
from themis.utils.logging_utils import configure_logging
from themis.utils.progress import ProgressReporter


def load_math_dataset(
    *,
    source: Literal["huggingface", "local"],
    data_dir: Path | None,
    limit: int | None,
    subjects: Sequence[str] | None,
    split: str = "test",
):
    """Load MATH-500 dataset.

    Args:
        source: Dataset source (huggingface or local)
        data_dir: Directory containing local dataset
        limit: Max rows to load
        subjects: Subjects to filter
        split: Dataset split

    Returns:
        List of generation examples
    """
    if source == "local":
        if data_dir is None:
            raise ValueError(
                "The --data-dir option is required when --source=local so Themis "
                "knows where to read the dataset."
            )
        samples = math500_dataset.load_math500(
            source="local",
            data_dir=data_dir,
            split=split,
            limit=limit,
            subjects=subjects,
        )
    else:
        samples = math500_dataset.load_math500(
            source="huggingface",
            split=split,
            limit=limit,
            subjects=subjects,
        )
    return [sample.to_generation_example() for sample in samples]


def load_competition_math_dataset(
    *,
    dataset: str,
    subset: str | None,
    source: Literal["huggingface", "local"],
    data_dir: Path | None,
    split: str,
    limit: int | None,
    subjects: Sequence[str] | None,
):
    """Load competition math dataset.

    Args:
        dataset: Dataset name
        subset: Dataset subset
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
    samples = competition_math_dataset.load_competition_math(
        dataset=dataset,
        subset=subset,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subjects,
    )
    return [sample.to_generation_example() for sample in samples]


def run_math_benchmark(
    rows: Sequence[dict[str, object]],
    *,
    max_samples: int | None,
    storage: Path | None,
    run_id: str | None,
    resume: bool,
    temperature: float,
    csv_output: Path | None,
    html_output: Path | None,
    json_output: Path | None,
    title: str,
    task_name: str,
) -> int:
    """Run math benchmark experiment.

    Args:
        rows: Dataset rows
        max_samples: Maximum samples to run
        storage: Cache directory
        run_id: Run identifier
        resume: Whether to resume from cache
        temperature: Sampling temperature
        csv_output: CSV export path
        html_output: HTML export path
        json_output: JSON export path
        title: Experiment title
        task_name: Task name

    Returns:
        Exit code
    """
    storage_impl = experiment_storage.ExperimentStorage(storage) if storage else None
    experiment = math_experiment.build_math500_zero_shot_experiment(
        temperature=temperature,
        storage=storage_impl,
        task_name=task_name,
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
    print(math_experiment.summarize_report(report))
    export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title=f"{title} experiment",
    )
    return 0


def math500_command(
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
    rows = load_math_dataset(
        source=source,
        data_dir=data_dir,
        limit=limit,
        subjects=subject_filter,
    )
    return run_math_benchmark(
        rows,
        max_samples=max_samples,
        storage=storage,
        run_id=run_id,
        resume=resume,
        temperature=temperature,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="math500",
        task_name="math500",
    )


def _create_competition_math_command(
    dataset_name: str,
    dataset_id: str,
    subset: str | None = None,
) -> Callable:
    """Create a competition math command function.

    Args:
        dataset_name: Display name for the dataset
        dataset_id: HuggingFace dataset ID
        subset: Optional dataset subset

    Returns:
        Command function
    """

    def command(
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
            tuple[str, ...], Parameter(help="Optional subject filters")
        ] = (),
        max_samples: Annotated[
            int | None, Parameter(help="Maximum samples to run")
        ] = None,
        storage: Annotated[
            Path | None, Parameter(help="Cache directory for datasets/results")
        ] = None,
        run_id: Annotated[
            str | None, Parameter(help="Identifier for cached run")
        ] = None,
        resume: Annotated[
            bool, Parameter(help="Reuse cached generations when storage is set")
        ] = True,
        temperature: Annotated[float, Parameter(help="Sampling temperature")] = 0.0,
        log_level: Annotated[
            str,
            Parameter(help="Logging level (critical/error/warning/info/debug/trace)"),
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
        f"""Run the {dataset_name} benchmark."""
        configure_logging(log_level)
        subject_filter = list(subjects) if subjects else None
        rows = load_competition_math_dataset(
            dataset=dataset_id,
            subset=subset,
            source=source,
            data_dir=data_dir,
            split=split,
            limit=limit,
            subjects=subject_filter,
        )

        return run_math_benchmark(
            rows,
            max_samples=max_samples,
            storage=storage,
            run_id=run_id,
            resume=resume,
            temperature=temperature,
            csv_output=csv_output,
            html_output=html_output,
            json_output=json_output,
            title=dataset_name,
            task_name=dataset_name,
        )

    command.__doc__ = f"Run the {dataset_name} benchmark."
    return command


# Create specific competition math commands
aime24_command = _create_competition_math_command("aime24", "math-ai/aime24")
aime25_command = _create_competition_math_command("aime25", "math-ai/aime25")
amc23_command = _create_competition_math_command("amc23", "math-ai/amc23")
olympiadbench_command = _create_competition_math_command(
    "olympiadbench", "math-ai/olympiadbench"
)
beyond_aime_command = _create_competition_math_command(
    "beyondaime", "ByteDance-Seed/BeyondAIME"
)
