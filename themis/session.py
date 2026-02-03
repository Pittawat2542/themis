"""Experiment session orchestration for vNext workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from themis.core.entities import ExperimentReport, ModelSpec, SamplingConfig
from themis.evaluation.pipeline import EvaluationPipelineContract
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.generation.plan import GenerationPlan
from themis.generation.router import ProviderRouter
from themis.generation.runner import GenerationRunner
from themis.generation.templates import PromptTemplate
from themis.interfaces import DatasetAdapter
from themis.presets import parse_model_name
from themis.providers import create_provider
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec


@dataclass
class ExperimentSession:
    """Main entry point for running experiments with vNext specs."""

    def run(
        self,
        spec: ExperimentSpec,
        *,
        execution: ExecutionSpec | None = None,
        storage: StorageSpec | None = None,
    ) -> ExperimentReport:
        execution = execution or ExecutionSpec()
        storage = storage or StorageSpec()

        pipeline = spec.pipeline
        if not isinstance(pipeline, EvaluationPipelineContract):
            raise TypeError(
                "spec.pipeline must implement EvaluationPipelineContract."
            )

        dataset = _resolve_dataset(spec.dataset)

        provider_name, model_id, provider_options = _parse_model(spec.model)
        model_spec = ModelSpec(identifier=model_id, provider=provider_name)
        sampling = _build_sampling(spec.sampling)

        plan = GenerationPlan(
            templates=[PromptTemplate(name="default", template=spec.prompt)],
            models=[model_spec],
            sampling_parameters=[sampling],
            dataset_id_field="id",
            reference_field="answer",
        )

        provider = create_provider(provider_name, **provider_options)
        router = ProviderRouter({(provider_name, model_id): provider})

        runner = GenerationRunner(
            provider=router,
            max_parallel=execution.workers,
            max_retries=execution.max_retries,
            retry_initial_delay=execution.retry_initial_delay,
            retry_backoff_multiplier=execution.retry_backoff_multiplier,
            retry_max_delay=execution.retry_max_delay,
            execution_backend=execution.backend,
        )

        storage_backend = _resolve_storage(storage)

        orchestrator = ExperimentOrchestrator(
            generation_plan=plan,
            generation_runner=runner,
            evaluation_pipeline=pipeline,
            storage=storage_backend,
        )

        return orchestrator.run(
            dataset=dataset,
            run_id=spec.run_id,
            resume=storage.cache,
            cache_results=storage.cache,
        )


def _parse_model(model: str) -> tuple[str, str, dict]:
    if ":" in model:
        provider_name, model_id = model.split(":", 1)
        return provider_name, model_id, {}
    return parse_model_name(model)


def _build_sampling(data: dict) -> SamplingConfig:
    return SamplingConfig(
        temperature=float(data.get("temperature", 0.0)),
        top_p=float(data.get("top_p", 0.95)),
        max_tokens=int(data.get("max_tokens", 512)),
    )


def _resolve_dataset(dataset: object) -> list[dict]:
    if isinstance(dataset, DatasetAdapter):
        return list(dataset.iter_samples())
    if isinstance(dataset, Iterable):
        return list(dataset)  # type: ignore[arg-type]
    raise TypeError("spec.dataset must be iterable or implement DatasetAdapter.")


def _resolve_storage(storage: StorageSpec):
    if storage.backend is not None:
        return storage.backend
    root = Path(storage.path) if storage.path is not None else Path(".cache/experiments")
    from themis.experiment.storage import ExperimentStorage

    return ExperimentStorage(root)


__all__ = ["ExperimentSession"]
