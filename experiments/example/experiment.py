"""Implementation for running the example experiment."""

from __future__ import annotations


from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import templates

from . import datasets as dataset_loader
from .config import ExampleExperimentConfig, ModelConfig


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

    report = built.orchestrator.run(
        dataset=dataset_rows,
        run_id=config.run_id,
        resume=config.resume,
    )
    return report


def summarize_report(report: orchestrator.ExperimentReport) -> str:
    exact = report.evaluation_report.metrics.get("ExactMatch")
    mean = exact.mean if exact else 0.0
    count = exact.count if exact else 0
    failures = len(report.failures) + len(report.evaluation_report.failures)
    return (
        f"Evaluated {report.metadata['total_samples']} samples | "
        f"exact match: {mean:.3f} over {count} generations | failures: {failures}"
    )


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


__all__ = ["run_experiment", "summarize_report"]
