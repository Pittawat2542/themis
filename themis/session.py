"""Experiment session orchestration for vNext workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from themis.core.entities import ExperimentReport, GenerationRecord, ModelSpec, SamplingConfig
from themis.evaluation.pipeline import EvaluationPipelineContract
from themis.experiment.manifest import build_reproducibility_manifest
from themis.experiment.cache_manager import CacheManager
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.generation.plan import GenerationPlan
from themis.generation.router import ProviderRouter
from themis.generation.runner import GenerationRunner
from themis.generation.strategies import RepeatedSamplingStrategy
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
        on_result: Callable[[GenerationRecord], None] | None = None,
    ) -> ExperimentReport:
        execution = execution or ExecutionSpec()
        storage = storage or StorageSpec()

        pipeline = spec.pipeline
        if not isinstance(pipeline, EvaluationPipelineContract):
            raise TypeError(
                "spec.pipeline must implement EvaluationPipelineContract."
            )

        dataset = _resolve_dataset(spec.dataset)

        provider_name, model_id, provider_options = _parse_model(
            spec.model, provider_options=spec.provider_options
        )
        model_spec = ModelSpec(identifier=model_id, provider=provider_name)
        sampling = _build_sampling(spec.sampling)

        plan = GenerationPlan(
            templates=[PromptTemplate(name="default", template=spec.prompt)],
            models=[model_spec],
            sampling_parameters=[sampling],
            dataset_id_field=spec.dataset_id_field,
            reference_field=spec.reference_field,
            metadata_fields=spec.metadata_fields,
        )

        provider = create_provider(provider_name, **provider_options)
        router = ProviderRouter({(provider_name, model_id): provider})

        strategy_resolver = None
        if spec.num_samples > 1:
            strategy_resolver = lambda task: RepeatedSamplingStrategy(
                attempts=spec.num_samples
            )

        runner = GenerationRunner(
            provider=router,
            strategy_resolver=strategy_resolver,
            max_parallel=execution.workers,
            max_retries=execution.max_retries,
            retry_initial_delay=execution.retry_initial_delay,
            retry_backoff_multiplier=execution.retry_backoff_multiplier,
            retry_max_delay=execution.retry_max_delay,
            execution_backend=execution.backend,
        )

        storage_backend = _resolve_storage(storage)
        cache_manager = CacheManager(
            storage=storage_backend,
            enable_resume=storage.cache,
            enable_cache=storage.cache,
        )
        manifest = build_reproducibility_manifest(
            model=model_id,
            provider=provider_name,
            provider_options=provider_options,
            sampling=dict(spec.sampling),
            num_samples=spec.num_samples,
            evaluation_config=_build_evaluation_config(pipeline),
            seeds={
                "provider_seed": provider_options.get("seed"),
                "sampling_seed": spec.sampling.get("seed"),
            },
        )

        orchestrator = ExperimentOrchestrator(
            generation_plan=plan,
            generation_runner=runner,
            evaluation_pipeline=pipeline,
            cache_manager=cache_manager,
        )

        return orchestrator.run(
            dataset=dataset,
            run_id=spec.run_id,
            resume=storage.cache,
            cache_results=storage.cache,
            on_result=on_result,
            run_manifest=manifest,
            max_records_in_memory=spec.max_records_in_memory,
        )


def _parse_model(
    model: str, *, provider_options: Mapping[str, Any] | None = None
) -> tuple[str, str, dict[str, Any]]:
    options = dict(provider_options or {})
    if ":" in model:
        provider_name, model_id = model.split(":", 1)
        return provider_name, model_id, options

    parsed_provider, model_id, parsed_options = parse_model_name(model, **options)
    return parsed_provider, model_id, parsed_options


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
        backend = storage.backend
        if hasattr(backend, "experiment_storage"):
            return backend.experiment_storage
        if not hasattr(backend, "start_run"):
            raise TypeError(
                "storage.backend must be ExperimentStorage-compatible."
            )
        return backend
    root = Path(storage.path) if storage.path is not None else Path(".cache/experiments")
    from themis.storage import ExperimentStorage

    return ExperimentStorage(root)


def _build_evaluation_config(pipeline: EvaluationPipelineContract) -> dict[str, Any]:
    if hasattr(pipeline, "evaluation_fingerprint"):
        try:
            fingerprint = dict(pipeline.evaluation_fingerprint())
        except Exception:
            fingerprint = {}
    else:
        fingerprint = {}

    if "metrics" not in fingerprint and hasattr(pipeline, "_metrics"):
        fingerprint["metrics"] = sorted(
            [
                f"{metric.__class__.__module__}.{metric.__class__.__name__}:{metric.name}"
                for metric in pipeline._metrics
            ]
        )
    if "extractor" not in fingerprint and hasattr(pipeline, "_extractor"):
        extractor = pipeline._extractor
        fingerprint["extractor"] = (
            f"{extractor.__class__.__module__}.{extractor.__class__.__name__}"
        )
        if "extractor_field" not in fingerprint and hasattr(extractor, "field_name"):
            fingerprint["extractor_field"] = extractor.field_name
    fingerprint.setdefault("metrics", [])
    fingerprint.setdefault("extractor", "unknown")

    return fingerprint


__all__ = ["ExperimentSession"]
