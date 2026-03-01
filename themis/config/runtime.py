"""Runtime helpers for executing experiments from Hydra configs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from themis.core import entities as core_entities
from themis.datasets import create_dataset
from themis.evaluation import extractors, metrics, pipeline as eval_pipeline
from themis.experiment import math as math_experiment
from themis.experiment import mcq as mcq_experiment
from themis.experiment import orchestrator as experiment_orchestrator
from themis import storage as experiment_storage
from themis.generation import plan, runner, templates
from themis.providers import registry as provider_registry

from . import registry, schema
from themis.exceptions import ConfigurationError


def _ensure_providers_registered() -> None:
    """Import provider modules to ensure they register themselves (lazy)."""
    try:
        from themis.providers import (
            litellm_provider,  # noqa: F401
            vllm_provider,  # noqa: F401
        )
    except ImportError:
        pass


def run_experiment_from_config(
    config: schema.ExperimentConfig,
    *,
    dataset: list[dict[str, object]] | None = None,
    on_result=None,
) -> experiment_orchestrator.ExperimentReport:
    _ensure_providers_registered()
    dataset_to_use = (
        dataset
        if dataset is not None
        else _load_dataset(config.dataset, experiment_name=config.name)
    )
    experiment = _build_experiment(config)
    return experiment.run(
        dataset_to_use,
        max_samples=config.max_samples,
        run_id=config.run_id,
        resume=config.resume,
        on_result=on_result,
    )


def summarize_report_for_config(
    config: schema.ExperimentConfig,
    report: experiment_orchestrator.ExperimentReport,
) -> str:
    if config.task in {
        "math500",
        "aime24",
        "aime25",
        "amc23",
        "olympiadbench",
        "beyondaime",
    }:
        return math_experiment.summarize_report(report)
    if config.task in {"supergpqa", "mmlu_pro"}:
        return mcq_experiment.summarize_report(report)
    raise ConfigurationError(f"Unsupported task '{config.task}' for summarization.")


def load_dataset_from_config(
    config: schema.ExperimentConfig,
) -> list[dict[str, object]]:
    return _load_dataset(config.dataset, experiment_name=config.name)


def _build_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    if config.task:
        builder = registry.get_experiment_builder(config.task)
        return builder(config)

    raise ConfigurationError(
        "Experiment configuration must specify a 'task'. "
        f"Available tasks: {', '.join(sorted(registry._EXPERIMENT_BUILDERS.keys()))}"
    )


def build_pipeline_from_config(
    config: schema.PipelineConfig,
) -> eval_pipeline.EvaluationPipeline:
    """Dynamically construct an evaluation pipeline from a YAML configuration."""

    # 1. Build Extractor
    extractor_obj = None
    if config.extractor:
        ext_name = config.extractor.name.lower()
        opts = config.extractor.options or {}
        if ext_name == "json_field":
            extractor_obj = extractors.JsonFieldExtractor(**opts)
        elif ext_name == "regex":
            extractor_obj = extractors.RegexExtractor(**opts)
        elif ext_name == "math_verify":
            extractor_obj = extractors.MathVerifyExtractor(**opts)
        elif ext_name == "identity":
            extractor_obj = extractors.IdentityExtractor(**opts)
        else:
            raise ConfigurationError(f"Unknown extractor: {ext_name}")
    else:
        extractor_obj = extractors.IdentityExtractor()

    # 2. Build Metrics
    metric_list = []
    for m_cfg in config.metrics:
        m_name = m_cfg.name.lower()
        opts = m_cfg.options or {}
        if m_name == "exact_match":
            metric_list.append(metrics.ExactMatch(**opts))
        elif m_name == "math_verify":
            metric_list.append(metrics.MathVerifyAccuracy(**opts))
        elif m_name == "response_length":
            metric_list.append(metrics.ResponseLength(**opts))
        elif m_name in ("rubric_judge", "llm_judge"):
            # Requires judge_model and judge_executor in options
            # If judge executor not specified, fallback to generation provider?
            if "judge_executor" in opts and isinstance(opts["judge_executor"], dict):
                p_cfg = opts.pop("judge_executor")
                j_prov = provider_registry.create_provider(
                    p_cfg.get("name", "fake"), **p_cfg.get("options", {})
                )
                opts["judge_executor"] = j_prov
            if "judge_model" in opts and isinstance(opts["judge_model"], dict):
                m_cfg_spec = opts.pop("judge_model")
                opts["judge_model"] = core_entities.ModelSpec(
                    identifier=m_cfg_spec.get("identifier", ""),
                    provider=m_cfg_spec.get("provider", ""),
                )
            metric_list.append(metrics.RubricJudgeMetric(**opts))
        else:
            # Try to resolve via metric_resolver
            from themis.evaluation.metric_resolver import resolve_metrics

            resolved = resolve_metrics([m_name])
            if resolved:
                # If it's a class we need to instantiate it with opts (hacky fallback)
                met = resolved[0]
                # resolve_metrics returns instances now, but we want to pass opts
                metric_list.append(met)
            else:
                raise ConfigurationError(f"Unknown metric: {m_name}")

    return eval_pipeline.EvaluationPipeline(
        extractor=extractor_obj, metrics=metric_list
    )


@registry.register_experiment_builder("math500")
@registry.register_experiment_builder("aime24")
@registry.register_experiment_builder("aime25")
@registry.register_experiment_builder("amc23")
@registry.register_experiment_builder("olympiadbench")
@registry.register_experiment_builder("beyondaime")
def _build_math_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    """Build a math evaluation experiment.

    Constructs an orchestrator and dataset for verifying mathematical reasoning
    using the `math_verify` extractor and exact-match metrics.

    Args:
        config: The experiment configuration.

    Returns:
        An instantiated `ExperimentOrchestrator`.
    """
    # Use the specific path if provided, otherwise use the default path
    storage_path = config.storage.path or config.storage.default_path
    storage = (
        experiment_storage.ExperimentStorage(Path(storage_path))
        if storage_path
        else None
    )
    sampling_cfg = core_entities.SamplingConfig(
        temperature=config.generation.sampling.temperature,
        top_p=config.generation.sampling.top_p,
        max_tokens=config.generation.sampling.max_tokens,
    )
    provider = provider_registry.create_provider(
        config.generation.provider.name, **config.generation.provider.options
    )
    runner_options = asdict(config.generation.runner)

    # Use the task name from config as the default task name
    task_name = config.task or "math500"
    # Override task name if provided in task_options
    if config.task_options and "task_name" in config.task_options:
        task_name = config.task_options["task_name"]

    return math_experiment.build_math500_zero_shot_experiment(
        model_client=provider,
        model_name=config.generation.model_identifier,
        storage=storage,
        sampling=sampling_cfg,
        provider_name=config.generation.provider.name,
        runner_options=runner_options,
        task_name=task_name,
    )


@registry.register_experiment_builder("custom")
def _build_custom_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    """Build a fully custom declarative experiment.

    Constructs an orchestrator driven entirely by YAML configuration,
    where pipelines, metrics, and models are resolved dynamically.

    Args:
        config: The experiment configuration.

    Returns:
        An instantiated `ExperimentOrchestrator`.
    """

    storage_path = config.storage.path or config.storage.default_path
    storage = (
        experiment_storage.ExperimentStorage(Path(storage_path))
        if storage_path
        else None
    )

    sampling_cfg = core_entities.SamplingConfig(
        temperature=config.generation.sampling.temperature,
        top_p=config.generation.sampling.top_p,
        max_tokens=config.generation.sampling.max_tokens,
    )

    provider = provider_registry.create_provider(
        config.generation.provider.name, **config.generation.provider.options
    )

    model_spec = core_entities.ModelSpec(
        identifier=config.generation.model_identifier,
        provider=config.generation.provider.name,
        default_sampling=sampling_cfg,
    )

    # Needs tasks templates from config.task_options
    template_str = config.task_options.get("template", "{question}")
    prompt_template = templates.PromptTemplate(
        name="custom-template",
        template=template_str,
    )

    gen_plan = plan.GenerationPlan(
        templates=[prompt_template],
        models=[model_spec],
        sampling_parameters=[sampling_cfg],
        dataset_id_field=config.task_options.get("dataset_id_field", "id"),
        reference_field=config.task_options.get("reference_field", "reference"),
    )

    runner_kwargs = config.generation.runner.__dict__
    gen_runner = runner.GenerationRunner(
        executor=provider,
        **runner_kwargs,
    )

    # Use custom pipeline if provided, else empty
    if config.pipeline:
        eval_pl = build_pipeline_from_config(config.pipeline)
    else:
        eval_pl = eval_pipeline.EvaluationPipeline(
            extractor=extractors.IdentityExtractor(), metrics=[]
        )

    from themis.experiment.cache_manager import CacheManager

    return experiment_orchestrator.ExperimentOrchestrator(
        generation_plan=gen_plan,
        generation_runner=gen_runner,
        evaluation_pipeline=eval_pl,
        cache_manager=CacheManager(
            storage=storage,
            enable_resume=config.resume,
            enable_cache=True,
        ),
    )


@registry.register_experiment_builder("supergpqa")
def _build_supergpqa_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    """Build a SuperGPQA multiple-choice experiment.

    Constructs an orchestrator and dataset tailored for the SuperGPQA benchmark,
    using letter extraction and exact-match metrics.

    Args:
        config: The experiment configuration.

    Returns:
        An instantiated `ExperimentOrchestrator`.
    """
    return _build_mcq_experiment(config, "supergpqa", "supergpqa")


@registry.register_experiment_builder("mmlu_pro")
def _build_mmlu_pro_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    """Build an MMLU-Pro multiple-choice experiment.

    Constructs an orchestrator and dataset tailored for the MMLU-Pro benchmark,
    using letter extraction and exact-match metrics.

    Args:
        config: The experiment configuration.

    Returns:
        An instantiated `ExperimentOrchestrator`.
    """
    return _build_mcq_experiment(config, "mmlu-pro", "mmlu_pro")


def _build_mcq_experiment(
    config: schema.ExperimentConfig, dataset_name: str, task_id: str
) -> experiment_orchestrator.ExperimentOrchestrator:
    """Build a generic multiple-choice question experiment.

    Constructs an orchestrator using a standard letter extraction pipeline
    (parsing A/B/C/D from generation text) and maps it against a specified
    dataset and task identifier.

    Args:
        config: The experiment configuration.
        dataset_name: The name of the dataset to load from the registry.
        task_id: The specific task identifier for the dataset slice.

    Returns:
        An instantiated `ExperimentOrchestrator`.
    """
    # Use the specific path if provided, otherwise use the default path
    storage_path = config.storage.path or config.storage.default_path
    storage = (
        experiment_storage.ExperimentStorage(Path(storage_path))
        if storage_path
        else None
    )
    sampling_cfg = core_entities.SamplingConfig(
        temperature=config.generation.sampling.temperature,
        top_p=config.generation.sampling.top_p,
        max_tokens=config.generation.sampling.max_tokens,
    )
    provider = provider_registry.create_provider(
        config.generation.provider.name, **config.generation.provider.options
    )
    runner_options = asdict(config.generation.runner)

    return mcq_experiment.build_multiple_choice_json_experiment(
        dataset_name=dataset_name,
        task_id=task_id,
        model_client=provider,
        model_name=config.generation.model_identifier,
        storage=storage,
        sampling=sampling_cfg,
        provider_name=config.generation.provider.name,
        runner_options=runner_options,
    )


def _load_dataset(
    config: schema.DatasetConfig, *, experiment_name: str
) -> List[dict[str, object]]:
    """Load dataset samples using the dataset registry.

    Args:
        config: Dataset configuration
        experiment_name: Name of the experiment (used to map to dataset)

    Returns:
        List of sample dictionaries ready for generation
    """
    # Handle inline datasets (not in registry)
    if config.source == "inline":
        if not config.inline_samples:
            raise ConfigurationError(
                "dataset.inline_samples must contain at least one row when"
                " dataset.source='inline'."
            )
        return list(config.inline_samples)

    # Use explicit dataset_id if provided
    dataset_name = config.dataset_id
    if not dataset_name:
        # Fallback to task name if dataset_id is not provided
        # This allows simple configs where task name matches dataset name
        # But we should probably enforce dataset_id for clarity in the future
        # For now, let's try to infer from task if available in config object passed to this function?
        # Wait, _load_dataset only gets DatasetConfig and experiment_name.
        # We should probably pass the full config or at least the task.
        # But for now, let's rely on dataset_id being present or raise error.
        raise ConfigurationError(
            "dataset.dataset_id must be provided when source is not 'inline'."
        )

    # Prepare options for dataset factory
    options = {
        "source": config.source,
        "data_dir": config.data_dir,
        "split": config.split,
        "limit": config.limit,
        "subjects": list(config.subjects) if config.subjects else None,
    }

    # Load samples via registry
    return create_dataset(dataset_name, **options)
