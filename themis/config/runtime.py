"""Runtime helpers for executing experiments from Hydra configs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from themis.core import entities as core_entities
from themis.datasets import math500 as math500_dataset
from themis.experiment import math as math_experiment
from themis.experiment import orchestrator as experiment_orchestrator
from themis.experiment import storage as experiment_storage
from themis.providers import registry as provider_registry

from . import schema

_SUPPORTED_EXPERIMENTS = {"math500_zero_shot"}


def run_experiment_from_config(
    config: schema.ExperimentConfig,
    *,
    dataset: list[dict[str, object]] | None = None,
    on_result=None,
) -> experiment_orchestrator.ExperimentReport:
    dataset_to_use = dataset if dataset is not None else _load_dataset(config.dataset)
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
    if config.name == "math500_zero_shot":
        return math_experiment.summarize_report(report)
    raise ValueError(
        f"Unsupported experiment '{config.name}'. Supported experiments:"
        f" {', '.join(sorted(_SUPPORTED_EXPERIMENTS))}."
    )


def load_dataset_from_config(
    config: schema.ExperimentConfig,
) -> list[dict[str, object]]:
    return _load_dataset(config.dataset)


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

    if config.name == "math500_zero_shot":
        return math_experiment.build_math500_zero_shot_experiment(
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


def _load_dataset(config: schema.DatasetConfig) -> List[dict[str, object]]:
    if config.source == "inline":
        if not config.inline_samples:
            raise ValueError(
                "dataset.inline_samples must contain at least one row when"
                " dataset.source='inline'."
            )
        return list(config.inline_samples)

    subjects = list(config.subjects) if config.subjects else None
    samples = math500_dataset.load_math500(
        source=config.source,
        data_dir=config.data_dir,
        limit=config.limit,
        subjects=subjects,
    )
    return [sample.to_generation_example() for sample in samples]
