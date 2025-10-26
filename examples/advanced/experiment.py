"""Advanced experiment runner showcasing extensibility."""

from __future__ import annotations


from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, strategies as evaluation_strategies
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import templates
from themis.utils.progress import ProgressReporter
from themis.generation import strategies as generation_strategies
from themis.project import ProjectExperiment

from experiments.example.config import ModelConfig

from . import datasets
from .config import AdvancedExperimentConfig
from .generation import PrioritizedGenerationRunner, TrackingProviderRouter
from .pipeline import SubjectAwareEvaluationPipeline


def create_project_experiment(config: AdvancedExperimentConfig) -> ProjectExperiment:
    """Create a project experiment from the advanced configuration."""
    
    prompt_text = _prompt_for_style(config.prompt_style)
    template = templates.PromptTemplate(
        name=f"advanced-{config.prompt_style}",
        template=prompt_text,
        metadata={"style": config.prompt_style},
    )

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject", "level", "dataset_name"),
        context_builder=lambda row: {"problem": row["problem"]},
    )
    
    # Create project experiment
    project_experiment = ProjectExperiment(
        name=f"advanced-math-experiment-{config.prompt_style}",
        description=f"Advanced math experiment using {config.prompt_style} prompt style",
        definition=definition,
        metadata={
            "storage_dir": config.storage_dir,
            "run_id": config.run_id,
            "resume": config.resume,
            "prompt_style": config.prompt_style,
            "test_time_attempts": config.test_time_attempts,
            "enable_subject_breakdown": config.enable_subject_breakdown,
        }
    )
    
    return project_experiment


def run_experiment(config: AdvancedExperimentConfig) -> orchestrator.ExperimentReport:
    rows = datasets.load_all_datasets(config.datasets)
    if not rows:
        raise ValueError("No dataset rows provided")

    prompt_text = _prompt_for_style(config.prompt_style)
    template = templates.PromptTemplate(
        name=f"advanced-{config.prompt_style}",
        template=prompt_text,
        metadata={"style": config.prompt_style},
    )

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject", "level", "dataset_name"),
        context_builder=lambda row: {"problem": row["problem"]},
    )

    strategy_resolver = _make_generation_strategy_resolver(config.test_time_attempts)
    evaluation_strategy = _make_evaluation_strategy_resolver(config.test_time_attempts)

    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),
        metrics=[metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)],
        runner_cls=PrioritizedGenerationRunner,
        runner_kwargs={"priority_field": "subject", "chunk_size": 2},
        pipeline_cls=SubjectAwareEvaluationPipeline,
        pipeline_kwargs={"subject_field": "subject"},
        router_cls=TrackingProviderRouter,
        strategy_resolver=strategy_resolver,
        evaluation_strategy_resolver=evaluation_strategy,
    )

    built = builder.build(definition, storage_dir=config.storage_dir)
    
    # Calculate total tasks for progress reporting
    total_tasks = 0
    for row in rows:
        # Each row will be processed for each model and sampling combination
        total_tasks += len(config.models) * len(config.samplings)

    with ProgressReporter(total=total_tasks, description="Generating") as progress:
        report = built.orchestrator.run(
            dataset=rows,
            run_id=config.run_id,
            resume=config.resume,
            on_result=progress.on_result,
        )

    if config.enable_subject_breakdown:
        report.metadata["subject_breakdown"] = built.pipeline.subject_breakdown
    report.metadata["generation_call_history"] = getattr(
        built.router, "call_history", []
    )

    return report


def summarize_subject_breakdown(report: orchestrator.ExperimentReport) -> str:
    breakdown = report.metadata.get("subject_breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}
    summary = (
        ", ".join(f"{subj}: {score:.2f}" for subj, score in breakdown.items()) or "n/a"
    )
    total = report.metadata["total_samples"]
    exact = report.evaluation_report.metrics.get("ExactMatch")
    mean = exact.mean if exact else 0.0
    return f"Samples={total}, ExactMatch={mean:.3f}, SubjectBreakdown={summary}"


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


def _prompt_for_style(style: str) -> str:
    if style == "cot":
        return (
            """
            You are a meticulous competition mathematician. Solve the problem step by step and
            respond with JSON containing fields `answer`, `reasoning`, and `confidence` (0-1).

            Problem:
            {problem}
            """
        ).strip()
    return (
        """
        Provide a concise answer to the problem below as JSON {{"answer": ..., "reasoning": "short"}}.

        Problem:
        {problem}
        """
    ).strip()


def _make_generation_strategy_resolver(attempts: int):
    if attempts <= 1:
        return lambda task: generation_strategies.SingleAttemptStrategy()
    return lambda task: generation_strategies.RepeatedSamplingStrategy(
        attempts=attempts
    )


def _make_evaluation_strategy_resolver(attempts: int):
    if attempts <= 1:
        return lambda record: evaluation_strategies.DefaultEvaluationStrategy()
    return lambda record: evaluation_strategies.AttemptAwareEvaluationStrategy()


__all__ = ["run_experiment", "summarize_subject_breakdown", "create_project_experiment"]