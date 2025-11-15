"""Cyclopts-powered CLI entrypoints for Themis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Callable, Iterable, Literal, Sequence

from cyclopts import App, Parameter

from themis.cli.new_project import create_project
from themis.config import (
    load_dataset_from_config,
    load_experiment_config,
    run_experiment_from_config,
    summarize_report_for_config,
)
from themis.datasets import (
    competition_math as competition_math_dataset,
    math500 as math500_dataset,
    mmlu_pro as mmlu_pro_dataset,
    super_gpqa as super_gpqa_dataset,
)
from themis.experiment import export as experiment_export
from themis.experiment import math as math_experiment
from themis.experiment import mcq as mcq_experiment
from themis.experiment import storage as experiment_storage
from themis.providers.registry import _REGISTRY
from themis.utils.logging_utils import configure_logging
from themis.utils.progress import ProgressReporter

# Import provider modules to ensure they register themselves
try:
    from themis.generation import clients  # noqa: F401 - registers fake provider
    from themis.generation.providers import litellm_provider  # noqa: F401
    from themis.generation.providers import vllm_provider  # noqa: F401
except ImportError:
    pass  # Some providers may not be available

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
    return _run_math_benchmark(
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


@app.command(name="aime24")
def aime24_cmd(
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
    """Run the AIME 2024 benchmark."""

    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = _load_competition_math_dataset(
        dataset="math-ai/aime24",
        subset=None,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    return _run_math_benchmark(
        rows,
        max_samples=max_samples,
        storage=storage,
        run_id=run_id,
        resume=resume,
        temperature=temperature,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="aime24",
        task_name="aime24",
    )


@app.command(name="aime25")
def aime25_cmd(
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
    """Run the AIME 2025 benchmark."""

    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = _load_competition_math_dataset(
        dataset="math-ai/aime25",
        subset=None,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    return _run_math_benchmark(
        rows,
        max_samples=max_samples,
        storage=storage,
        run_id=run_id,
        resume=resume,
        temperature=temperature,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="aime25",
        task_name="aime25",
    )


@app.command(name="amc23")
def amc23_cmd(
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
    """Run the AMC 2023 benchmark."""

    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = _load_competition_math_dataset(
        dataset="math-ai/amc23",
        subset=None,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    return _run_math_benchmark(
        rows,
        max_samples=max_samples,
        storage=storage,
        run_id=run_id,
        resume=resume,
        temperature=temperature,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="amc23",
        task_name="amc23",
    )


@app.command(name="olympiadbench")
def olympiadbench_cmd(
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
    """Run the OlympiadBench benchmark."""

    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = _load_competition_math_dataset(
        dataset="math-ai/olympiadbench",
        subset=None,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    return _run_math_benchmark(
        rows,
        max_samples=max_samples,
        storage=storage,
        run_id=run_id,
        resume=resume,
        temperature=temperature,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="olympiadbench",
        task_name="olympiadbench",
    )


@app.command(name="beyondaime")
def beyond_aime_cmd(
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
    """Run the BeyondAIME benchmark."""

    configure_logging(log_level)
    subject_filter = list(subjects) if subjects else None
    rows = _load_competition_math_dataset(
        dataset="ByteDance-Seed/BeyondAIME",
        subset=None,
        source=source,
        data_dir=data_dir,
        split=split,
        limit=limit,
        subjects=subject_filter,
    )

    return _run_math_benchmark(
        rows,
        max_samples=max_samples,
        storage=storage,
        run_id=run_id,
        resume=resume,
        temperature=temperature,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="beyondaime",
        task_name="beyondaime",
    )


@app.command(name="supergpqa")
def supergpqa_cmd(
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
    rows = _load_multiple_choice_dataset(
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

    total = _effective_total(len(rows), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            rows,
            max_samples=max_samples,
            run_id=run_id,
            resume=resume,
            on_result=progress.on_result,
        )
    print(mcq_experiment.summarize_report(report))
    _export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="supergpqa experiment",
    )
    return 0


@app.command(name="mmlu-pro")
def mmlu_pro_cmd(
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
    rows = _load_multiple_choice_dataset(
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

    total = _effective_total(len(rows), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            rows,
            max_samples=max_samples,
            run_id=run_id,
            resume=resume,
            on_result=progress.on_result,
        )
    print(mcq_experiment.summarize_report(report))
    _export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="mmlu_pro experiment",
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
    split: str = "test",
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


def _load_multiple_choice_dataset(
    *,
    loader: Callable[..., Sequence],
    source: Literal["huggingface", "local"],
    data_dir: Path | None,
    split: str,
    limit: int | None,
    subjects: Iterable[str] | None,
):
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


def _load_competition_math_dataset(
    *,
    dataset: str,
    subset: str | None,
    source: Literal["huggingface", "local"],
    data_dir: Path | None,
    split: str,
    limit: int | None,
    subjects: Iterable[str] | None,
):
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


def _effective_total(total: int, limit: int | None) -> int:
    if limit is None:
        return total
    return min(total, limit)


def _run_math_benchmark(
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
    storage_impl = experiment_storage.ExperimentStorage(storage) if storage else None
    experiment = math_experiment.build_math500_zero_shot_experiment(
        temperature=temperature,
        storage=storage_impl,
        task_name=task_name,
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
        title=f"{title} experiment",
    )
    return 0


@app.command(name="list-providers")
def list_providers(
    *,
    verbose: Annotated[
        bool, Parameter(help="Show detailed provider information")
    ] = False,
) -> int:
    """List available LLM providers."""

    providers = sorted(_REGISTRY._factories.keys())

    if not providers:
        print("No providers registered.")
        return 0

    print("Available Providers:")
    print("=" * 60)

    provider_info = {
        "fake": "Built-in fake provider for testing (no API required)",
        "openai-compatible": "OpenAI-compatible API (LM Studio, Ollama, vLLM, OpenAI)",
        "vllm": "vLLM server provider for local model hosting",
    }

    for provider in providers:
        status = "✓" if provider in provider_info else "·"
        print(f"{status} {provider}")
        if verbose and provider in provider_info:
            print(f"  {provider_info[provider]}")

    if not verbose:
        print("\nUse --verbose for more details")

    return 0


@app.command(name="list-benchmarks")
def list_benchmarks(
    *,
    verbose: Annotated[
        bool, Parameter(help="Show detailed benchmark information")
    ] = False,
) -> int:
    """List available datasets and benchmarks."""

    benchmarks = [
        {
            "name": "math500",
            "description": "MATH-500 dataset for mathematical reasoning",
            "source": "huggingface (default) or local",
            "subjects": [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ],
            "command": "uv run python -m themis.cli math500",
        },
        {
            "name": "supergpqa",
            "description": "Graduate-level QA benchmark with multiple-choice questions",
            "source": "huggingface (default) or local",
            "subjects": "category filter via --subjects",
            "command": "uv run python -m themis.cli supergpqa",
        },
        {
            "name": "mmlu-pro",
            "description": "Professional-level MMLU benchmark with refined distractors",
            "source": "huggingface (default) or local",
            "subjects": "subject filter via --subjects",
            "command": "uv run python -m themis.cli mmlu-pro",
        },
        {
            "name": "aime24",
            "description": "AIME 2024 competition problems",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli aime24",
        },
        {
            "name": "aime25",
            "description": "AIME 2025 competition problems",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli aime25",
        },
        {
            "name": "amc23",
            "description": "AMC 2023 competition problems",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli amc23",
        },
        {
            "name": "olympiadbench",
            "description": "Mixed Olympiad-style math benchmark",
            "source": "huggingface (default) or local",
            "subjects": "competition metadata",
            "command": "uv run python -m themis.cli olympiadbench",
        },
        {
            "name": "beyondaime",
            "description": "BeyondAIME advanced math competition set",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli beyondaime",
        },
        {
            "name": "demo",
            "description": "Built-in demo with 2 math problems",
            "source": "inline",
            "subjects": ["precalculus", "arithmetic"],
            "command": "uv run python -m themis.cli demo",
        },
        {
            "name": "inline",
            "description": "Custom inline dataset (via config file)",
            "source": "config file",
            "subjects": "user-defined",
            "command": "uv run python -m themis.cli run-config --config your_config.yaml",
        },
    ]

    print("Available Datasets & Benchmarks:")
    print("=" * 60)

    for bench in benchmarks:
        print(f"\n📊 {bench['name']}")
        print(f"   {bench['description']}")
        if verbose:
            print(f"   Source: {bench['source']}")
            if isinstance(bench["subjects"], list):
                print(f"   Subjects: {', '.join(bench['subjects'])}")
            else:
                print(f"   Subjects: {bench['subjects']}")
            print(f"   Command: {bench['command']}")

    if not verbose:
        print("\nUse --verbose for more details and example commands")

    return 0


@app.command(name="validate-config")
def validate_config(
    *,
    config: Annotated[Path, Parameter(help="Path to config file to validate")],
) -> int:
    """Validate a configuration file without running the experiment."""

    if not config.exists():
        print(f"❌ Error: Config file not found: {config}")
        return 1

    print(f"Validating config: {config}")
    print("-" * 60)

    try:
        # Try to load as experiment config
        experiment_config = load_experiment_config(config, overrides=())
        print("✓ Config file is valid")
        print(f"\nExperiment: {experiment_config.name}")
        print(f"Run ID: {experiment_config.run_id or '(auto-generated)'}")
        print(f"Resume: {experiment_config.resume}")
        print(f"Max samples: {experiment_config.max_samples or '(unlimited)'}")

        print(f"\nDataset:")
        print(f"  Source: {experiment_config.dataset.source}")
        print(f"  Split: {experiment_config.dataset.split}")
        if experiment_config.dataset.limit:
            print(f"  Limit: {experiment_config.dataset.limit}")
        if experiment_config.dataset.subjects:
            print(f"  Subjects: {', '.join(experiment_config.dataset.subjects)}")

        print(f"\nGeneration:")
        print(f"  Model: {experiment_config.generation.model_identifier}")
        print(f"  Provider: {experiment_config.generation.provider.name}")
        print(f"  Temperature: {experiment_config.generation.sampling.temperature}")
        print(f"  Max tokens: {experiment_config.generation.sampling.max_tokens}")

        if experiment_config.storage.path:
            print(f"\nStorage: {experiment_config.storage.path}")

        return 0
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return 1


@app.command(name="info")
def show_info() -> int:
    """Show system information and installed components."""

    import themis
    from themis import _version

    print("Themis Information")
    print("=" * 60)
    print(f"Version: {getattr(_version, '__version__', 'unknown')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    print("\n📦 Installed Providers:")
    providers = sorted(_REGISTRY._factories.keys())
    for provider in providers:
        print(f"  ✓ {provider}")

    print("\n📊 Available Benchmarks:")
    benchmarks = [
        "demo",
        "math500",
        "aime24",
        "aime25",
        "amc23",
        "olympiadbench",
        "beyondaime",
        "supergpqa",
        "mmlu-pro",
        "inline (via config)",
    ]
    for bench in benchmarks:
        print(f"  ✓ {bench}")

    print("\n📁 Example Locations:")
    examples_dir = Path(themis.__file__).parent.parent / "examples"
    if examples_dir.exists():
        print(f"  {examples_dir}")
        example_dirs = sorted(
            [
                d.name
                for d in examples_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ]
        )
        for ex in example_dirs:
            print(f"    • {ex}/")

    print("\n📚 Documentation:")
    print("  examples/README.md - Comprehensive tutorial cookbook")
    print("  COOKBOOK.md - Quick reference guide")
    print("  docs/ - Detailed documentation")

    print("\n🚀 Quick Start:")
    print("  uv run python -m themis.cli demo")
    print("  uv run python -m themis.cli list-providers")
    print("  uv run python -m themis.cli list-benchmarks")

    return 0


@app.command(name="init")
def init_config(
    *,
    output: Annotated[Path, Parameter(help="Output path for config file")] = Path(
        "themis_config.yaml"
    ),
    template: Annotated[
        Literal["basic", "math500", "inline"],
        Parameter(help="Config template to generate"),
    ] = "basic",
) -> int:
    """Generate a sample configuration file for use with run-config."""

    templates = {
        "basic": """name: my_experiment
dataset:
  source: huggingface
  limit: 50
generation:
  model_identifier: fake-math-llm
  provider:
    name: fake
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
  runner:
    max_parallel: 1
    max_retries: 3
storage:
  path: .cache/my_experiment
run_id: my-experiment-001
resume: true
""",
        "math500": """name: math500_evaluation
dataset:
  source: huggingface
  limit: null  # No limit, run full dataset
  subjects:
    - algebra
    - geometry
generation:
  model_identifier: my-model
  provider:
    name: openai-compatible
    options:
      base_url: http://localhost:1234/v1
      api_key: not-needed
      model_name: qwen2.5-7b-instruct
      timeout: 60
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
  runner:
    max_parallel: 4
    max_retries: 3
    retry_initial_delay: 0.5
    retry_backoff_multiplier: 2.0
    retry_max_delay: 2.0
storage:
  path: .cache/math500
run_id: math500-run-001
resume: true
max_samples: null
""",
        "inline": """name: inline_dataset_experiment
dataset:
  source: inline
  inline_samples:
    - unique_id: sample-1
      problem: "What is 2 + 2?"
      answer: "4"
      subject: arithmetic
      level: 1
    - unique_id: sample-2
      problem: "Solve for x: 2x + 5 = 13"
      answer: "4"
      subject: algebra
      level: 2
generation:
  model_identifier: fake-math-llm
  provider:
    name: fake
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
storage:
  path: .cache/inline_experiment
run_id: inline-001
resume: true
""",
    }

    if output.exists():
        print(f"❌ Error: File already exists: {output}")
        print("   Use a different --output path or delete the existing file")
        return 1

    config_content = templates[template]

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(config_content)

        print(f"✓ Created config file: {output}")
        print(f"  Template: {template}")
        print("\n📝 Next steps:")
        print(f"  1. Edit {output} to customize settings")
        print(
            f"  2. Validate: uv run python -m themis.cli validate-config --config {output}"
        )
        print(f"  3. Run: uv run python -m themis.cli run-config --config {output}")

        if template == "math500":
            print("\n⚠️  Remember to:")
            print("  • Update provider.options.base_url with your LLM server endpoint")
            print("  • Update provider.options.model_name with your actual model")
            print("  • Set provider.options.api_key if required by your server")
        elif template == "inline":
            print("\n💡 Tip:")
            print("  • Add more samples to dataset.inline_samples list")
            print("  • Each sample needs: unique_id, problem, answer")

        return 0
    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return 1


@app.command(name="new-project")
def new_project(
    *,
    project_name: Annotated[str, Parameter(help="The name of the new project")],
    project_path: Annotated[
        Path,
        Parameter(help="The path where the new project will be created"),
    ] = Path("."),
) -> int:
    """Create a new Themis project."""
    try:
        create_project(project_name, project_path)
        print(f"Successfully created new project '{project_name}' in {project_path}")
        return 0
    except FileExistsError as e:
        print(f"Error: {e}")
        return 1


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
