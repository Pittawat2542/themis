"""Agentic experiment wiring."""

from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import templates

from experiments.example import datasets as base_datasets

from .config import AgenticExperimentConfig
from .runner import AgenticRunner


def run_experiment(config: AgenticExperimentConfig) -> orchestrator.ExperimentReport:
    rows: list[dict[str, object]] = []
    for dataset_cfg in config.datasets:
        rows.extend(base_datasets.load_dataset(dataset_cfg))
    if not rows:
        raise ValueError("Agentic experiment requires dataset rows")

    template = templates.PromptTemplate(
        name="agentic-zero-shot",
        template="""
        Solve the following math problem and respond with JSON {{"answer": ..., "reasoning": "short"}}.

        Problem:
        {problem}
        """.strip(),
        metadata={"task": "agentic"},
    )

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject", "level"),
        context_builder=lambda row: {"problem": row["problem"]},
    )

    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),
        metrics=[
            metrics.ExactMatch(case_sensitive=False, strip_whitespace=True),
            metrics.ResponseLength(),
        ],
        runner_cls=AgenticRunner,
        runner_kwargs={
            "planner_prompt": config.planner_prompt,
            "final_prompt_prefix": config.final_prompt_prefix,
        },
    )

    built = builder.build(definition, storage_dir=config.storage_dir)
    report = built.orchestrator.run(
        dataset=rows,
        run_id=config.run_id,
        resume=config.resume,
    )
    return report


def _make_binding(model_cfg) -> experiment_builder.ModelBinding:
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


__all__ = ["run_experiment"]
