"""Utilities for assembling experiments from reusable components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Type

from themis.config import schema as config
from themis.core import entities as core_entities
from themis.evaluation import pipeline as evaluation_pipeline
from themis.evaluation import strategies as evaluation_strategies
from themis.experiment import orchestrator
from themis.experiment import storage as experiment_storage
from themis.experiment.cache_manager import CacheManager
from themis.experiment.definitions import (
    BuiltExperiment,
    ExperimentDefinition,
    ModelBinding,
)
from themis.experiment.integration_manager import IntegrationManager
from themis.generation import plan as generation_plan
from themis.generation import router as generation_router
from themis.generation import runner as generation_runner
from themis.generation import strategies as generation_strategies
from themis.interfaces import ModelProvider
from themis.providers import create_provider


class ExperimentBuilder:
    """Composable builder for constructing experiment components."""

    def __init__(
        self,
        *,
        extractor,
        metrics,
        runner_cls: Type[
            generation_runner.GenerationRunner
        ] = generation_runner.GenerationRunner,
        runner_kwargs: Mapping[str, Any] | None = None,
        pipeline_cls: Type[
            evaluation_pipeline.EvaluationPipeline
        ] = evaluation_pipeline.EvaluationPipeline,
        pipeline_kwargs: Mapping[str, Any] | None = None,
        router_cls: Type[ModelProvider] = generation_router.ProviderRouter,
        router_kwargs: Mapping[str, Any] | None = None,
        strategy_resolver: Callable[
            [core_entities.GenerationTask], generation_strategies.GenerationStrategy
        ]
        | None = None,
        evaluation_strategy_resolver: Callable[
            [core_entities.GenerationRecord], evaluation_strategies.EvaluationStrategy
        ]
        | None = None,
    ) -> None:
        self._extractor = extractor
        self._metrics = list(metrics)
        self._runner_cls = runner_cls
        self._runner_kwargs = dict(runner_kwargs or {})
        self._pipeline_cls = pipeline_cls
        self._pipeline_kwargs = dict(pipeline_kwargs or {})
        self._router_cls = router_cls
        self._router_kwargs = dict(router_kwargs or {})
        self._strategy_resolver = strategy_resolver
        self._evaluation_strategy_resolver = evaluation_strategy_resolver

    def build(
        self,
        definition: ExperimentDefinition,
        *,
        storage_dir: str | Path | None = None,
    ) -> BuiltExperiment:
        plan_obj = self._build_plan(definition)
        router = self._build_router(definition.model_bindings)
        runner_kwargs = dict(self._runner_kwargs)
        if self._strategy_resolver is not None:
            runner_kwargs.setdefault("strategy_resolver", self._strategy_resolver)
        runner = self._runner_cls(provider=router, **runner_kwargs)
        pipeline_kwargs = dict(self._pipeline_kwargs)
        if self._evaluation_strategy_resolver is not None:
            pipeline_kwargs.setdefault(
                "strategy_resolver", self._evaluation_strategy_resolver
            )
        pipeline = self._pipeline_cls(
            extractor=self._extractor,
            metrics=self._metrics,
            **pipeline_kwargs,
        )

        # Create storage backend
        storage = (
            experiment_storage.ExperimentStorage(storage_dir)
            if storage_dir is not None
            else None
        )

        # Create managers for better separation of concerns
        cache_manager = CacheManager(
            storage=storage,
            enable_resume=True,
            enable_cache=True,
        )
        integration_manager = IntegrationManager(config=config.IntegrationsConfig())

        # Create orchestrator with managers
        orchestrator_obj = orchestrator.ExperimentOrchestrator(
            generation_plan=plan_obj,
            generation_runner=runner,
            evaluation_pipeline=pipeline,
            cache_manager=cache_manager,
            integration_manager=integration_manager,
        )

        return BuiltExperiment(
            orchestrator=orchestrator_obj,
            plan=plan_obj,
            runner=runner,
            pipeline=pipeline,
            storage=storage,
            router=router,
        )

    def _build_plan(
        self, definition: ExperimentDefinition
    ) -> generation_plan.GenerationPlan:
        return generation_plan.GenerationPlan(
            templates=list(definition.templates),
            models=[binding.spec for binding in definition.model_bindings],
            sampling_parameters=list(definition.sampling_parameters),
            dataset_id_field=definition.dataset_id_field,
            reference_field=definition.reference_field,
            metadata_fields=tuple(definition.metadata_fields),
            context_builder=definition.context_builder,
        )

    def _build_router(self, bindings: Sequence[ModelBinding]) -> ModelProvider:
        providers: dict[str, ModelProvider] = {}
        for binding in bindings:
            providers[binding.spec.identifier] = create_provider(
                binding.provider_name,
                **binding.provider_options,
            )
        return self._router_cls(providers, **self._router_kwargs)


__all__ = [
    "ExperimentBuilder",
    "ExperimentDefinition",
    "ModelBinding",
    "BuiltExperiment",
]
