"""Runtime helpers for executing experiments from Hydra configs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from themis.core import entities as core_entities
from themis.datasets import create_dataset
from themis.experiment import math as math_experiment
from themis.experiment import mcq as mcq_experiment
from themis.experiment import orchestrator as experiment_orchestrator
from themis.experiment import storage as experiment_storage
from themis.providers import registry as provider_registry

from . import registry, schema




def run_experiment_from_config(
    config: schema.ExperimentConfig,
    *,
    dataset: list[dict[str, object]] | None = None,
    on_result=None,
) -> experiment_orchestrator.ExperimentReport:
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
    raise ValueError(f"Unsupported task '{config.task}' for summarization.")


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

    raise ValueError(
        "Experiment configuration must specify a 'task'. "
        f"Available tasks: {', '.join(sorted(registry._EXPERIMENT_BUILDERS.keys()))}"
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


@registry.register_experiment_builder("supergpqa")
def _build_supergpqa_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    return _build_mcq_experiment(config, "supergpqa", "supergpqa")


@registry.register_experiment_builder("mmlu_pro")
def _build_mmlu_pro_experiment(
    config: schema.ExperimentConfig,
) -> experiment_orchestrator.ExperimentOrchestrator:
    return _build_mcq_experiment(config, "mmlu-pro", "mmlu_pro")


def _build_mcq_experiment(
    config: schema.ExperimentConfig, dataset_name: str, task_id: str
) -> experiment_orchestrator.ExperimentOrchestrator:
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
            raise ValueError(
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
        raise ValueError(
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
