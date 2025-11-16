"""Runtime helpers for executing experiments from Hydra configs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from themis.core import entities as core_entities
from themis.datasets import create_dataset, list_datasets
from themis.experiment import math as math_experiment
from themis.experiment import mcq as mcq_experiment
from themis.experiment import orchestrator as experiment_orchestrator
from themis.experiment import storage as experiment_storage
from themis.providers import registry as provider_registry

from . import schema

_COMPETITION_EXPERIMENTS = {
    "aime24_zero_shot": {"dataset": "math-ai/aime24", "task": "aime24"},
    "aime25_zero_shot": {"dataset": "math-ai/aime25", "task": "aime25"},
    "amc23_zero_shot": {"dataset": "math-ai/amc23", "task": "amc23"},
    "olympiadbench_zero_shot": {
        "dataset": "math-ai/olympiadbench",
        "task": "olympiadbench",
    },
    "beyondaime_zero_shot": {
        "dataset": "ByteDance-Seed/BeyondAIME",
        "task": "beyondaime",
    },
}

# Mapping from experiment name to dataset registry name
_EXPERIMENT_TO_DATASET = {
    "math500_zero_shot": "math500",
    "supergpqa_zero_shot": "supergpqa",
    "mmlu_pro_zero_shot": "mmlu-pro",
    # Competition experiments map to specific dataset aliases
    "aime24_zero_shot": "aime24",
    "aime25_zero_shot": "aime25",
    "amc23_zero_shot": "amc23",
    "olympiadbench_zero_shot": "olympiadbench",
    "beyondaime_zero_shot": "beyondaime",
}

_SUPPORTED_EXPERIMENTS = set(_EXPERIMENT_TO_DATASET.keys())

_MATH_EXPERIMENT_NAMES = {"math500_zero_shot"} | set(_COMPETITION_EXPERIMENTS.keys())


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
    if config.name in _MATH_EXPERIMENT_NAMES:
        return math_experiment.summarize_report(report)
    if config.name in {"supergpqa_zero_shot", "mmlu_pro_zero_shot"}:
        return mcq_experiment.summarize_report(report)
    raise ValueError(
        f"Unsupported experiment '{config.name}'. Supported experiments:"
        f" {', '.join(sorted(_SUPPORTED_EXPERIMENTS))}."
    )


def load_dataset_from_config(
    config: schema.ExperimentConfig,
) -> list[dict[str, object]]:
    return _load_dataset(config.dataset, experiment_name=config.name)


def _build_experiment(
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

    if config.name in _MATH_EXPERIMENT_NAMES:
        task_name = _COMPETITION_EXPERIMENTS.get(config.name, {}).get("task", "math500")
        return math_experiment.build_math500_zero_shot_experiment(
            model_client=provider,
            model_name=config.generation.model_identifier,
            storage=storage,
            sampling=sampling_cfg,
            provider_name=config.generation.provider.name,
            runner_options=runner_options,
            task_name=task_name,
        )
    if config.name == "supergpqa_zero_shot":
        return mcq_experiment.build_multiple_choice_json_experiment(
            dataset_name="supergpqa",
            task_id="supergpqa",
            model_client=provider,
            model_name=config.generation.model_identifier,
            storage=storage,
            sampling=sampling_cfg,
            provider_name=config.generation.provider.name,
            runner_options=runner_options,
        )
    if config.name == "mmlu_pro_zero_shot":
        return mcq_experiment.build_multiple_choice_json_experiment(
            dataset_name="mmlu-pro",
            task_id="mmlu_pro",
            model_client=provider,
            model_name=config.generation.model_identifier,
            storage=storage,
            sampling=sampling_cfg,
            provider_name=config.generation.provider.name,
            runner_options=runner_options,
        )

    raise ValueError(
        f"Unsupported experiment '{config.name}'. Supported experiments:"
        f" {', '.join(sorted(_SUPPORTED_EXPERIMENTS))}."
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

    # Map experiment name to dataset registry name
    dataset_name = _EXPERIMENT_TO_DATASET.get(experiment_name)
    if not dataset_name:
        raise ValueError(
            f"Unsupported experiment '{experiment_name}'. Supported experiments:"
            f" {', '.join(sorted(_SUPPORTED_EXPERIMENTS))}."
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
