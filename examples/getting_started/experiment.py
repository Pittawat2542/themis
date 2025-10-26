"""Implementation for running the example experiment."""

from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import templates
from themis.project import ProjectExperiment
from themis.utils.progress import ProgressReporter

from . import datasets as dataset_loader
from .config import ExampleExperimentConfig, ModelConfig


def create_project_experiment(config: ExampleExperimentConfig) -> ProjectExperiment:
    """Create a project experiment from the example configuration."""

    template = templates.PromptTemplate(
        name="math-zero-shot",
        template="""
        You are an expert mathematician. Solve the problem below and respond with a JSON object
        containing `answer` and `reasoning` keys only.

        Problem:
        {problem}
        """.strip(),
        metadata={"task": "example"},
    )

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("dataset_name", "subject", "level"),
        context_builder=lambda row: {"problem": row["problem"]},
    )

    # Create project experiment
    project_experiment = ProjectExperiment(
        name="example-math-experiment",
        description="Example math experiment using JSON extraction",
        definition=definition,
        metadata={
            "storage_dir": config.storage_dir,
            "run_id": config.run_id,
            "resume": config.resume,
        },
    )

    return project_experiment


def run_experiment(config: ExampleExperimentConfig) -> orchestrator.ExperimentReport:
    dataset_rows: list[dict[str, object]] = []
    for dataset_cfg in config.datasets:
        rows = dataset_loader.load_dataset(dataset_cfg)
        dataset_rows.extend(rows)

    if not dataset_rows:
        raise ValueError("Example experiment requires at least one dataset row")

    template = templates.PromptTemplate(
        name="math-zero-shot",
        template="""
        You are an expert mathematician. Solve the problem below and respond with a JSON object
        containing `answer` and `reasoning` keys only.

        Problem:
        {problem}
        """.strip(),
        metadata={"task": "example"},
    )

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("dataset_name", "subject", "level"),
        context_builder=lambda row: {"problem": row["problem"]},
    )

    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),
        metrics=[metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)],
    )

    built = builder.build(definition, storage_dir=config.storage_dir)

    # Calculate total tasks for progress reporting
    total_tasks = 0
    for row in dataset_rows:
        # Each row will be processed for each model and sampling combination
        total_tasks += len(config.models) * len(config.samplings)

    with ProgressReporter(total=total_tasks, description="Generating") as progress:
        report = built.orchestrator.run(
            dataset=dataset_rows,
            run_id=config.run_id,
            resume=config.resume,
            on_result=progress.on_result,
        )
    return report


def summarize_report(report: orchestrator.ExperimentReport) -> str:
    # Get exact match metric
    exact = report.evaluation_report.metrics.get("ExactMatch")
    exact_mean = exact.mean if exact else 0.0
    exact_count = exact.count if exact else 0

    # Get failure counts
    generation_failures = len(report.failures)
    evaluation_failures = len(report.evaluation_report.failures)
    total_failures = generation_failures + evaluation_failures

    # Get metadata
    total_samples = report.metadata.get("total_samples", 0)
    successful_generations = report.metadata.get("successful_generations", 0)
    failed_generations = report.metadata.get("failed_generations", 0)

    # Build summary string
    summary_parts = [
        f"Evaluated {total_samples} samples",
        f"Successful generations: {successful_generations}/{total_samples}",
        f"Exact match: {exact_mean:.3f} ({exact_count} evaluated)",
    ]

    # Add failure information
    if total_failures > 0:
        summary_parts.append(
            f"Failures: {total_failures} (gen: {failed_generations}, eval: {evaluation_failures})"
        )
    else:
        summary_parts.append("No failures")

    return " | ".join(summary_parts)


def _make_binding(model_cfg: ModelConfig) -> experiment_builder.ModelBinding:
    spec = core_entities.ModelSpec(
        identifier=model_cfg.name,
        provider=model_cfg.provider,
        metadata={"description": model_cfg.description or model_cfg.name},
    )
    return experiment_builder.ModelBinding(
        spec=spec,
        provider_name=model_cfg.provider,
        provider_options=model_cfg.provider_options,
    )


__all__ = ["run_experiment", "summarize_report", "create_project_experiment"]
