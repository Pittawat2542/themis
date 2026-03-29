"""Experiment authoring model and snapshot compilation for Phase 3."""

from __future__ import annotations

import asyncio

from pydantic import Field

from themis.core.builtins import (
    resolve_generator_component,
    resolve_judge_model_component,
    resolve_metric_component,
    resolve_parser_component,
    resolve_reducer_component,
)
from themis.core.base import FrozenModel
from themis.core.components import component_ref_from_value
from themis.core.config import EvaluationConfig, GenerationConfig, RuntimeConfig, StorageConfig
from themis.core.models import Dataset
from themis.core.orchestrator import Orchestrator
from themis.core.protocols import LLMMetric, LifecycleSubscriber, PureMetric, SelectionMetric, TraceMetric, TracingProvider
from themis.core.store import RunStore
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.sqlite import sqlite_store
from themis.core.tracing import NoOpTracingProvider
from themis.core.snapshot import (
    ComponentRefs,
    DatasetRef,
    RunIdentity,
    RunProvenance,
    RunSnapshot,
)


class Experiment(FrozenModel):
    """Compiled-input experiment definition for Themis v4."""

    generation: GenerationConfig
    evaluation: EvaluationConfig
    storage: StorageConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    datasets: list[Dataset] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    environment_metadata: dict[str, str] = Field(default_factory=dict)
    themis_version: str = "4.0.0a0"
    python_version: str = "3.12"
    platform: str = "unknown"

    def compile(self) -> RunSnapshot:
        return self._compile_with_runtime(self.runtime)

    def _compile_with_runtime(self, runtime: RuntimeConfig) -> RunSnapshot:
        component_refs = ComponentRefs(
            generator=component_ref_from_value(self.generation.generator),
            reducer=component_ref_from_value(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=[component_ref_from_value(parser) for parser in self.evaluation.parsers],
            metrics=[component_ref_from_value(metric) for metric in self.evaluation.metrics],
            judge_models=[component_ref_from_value(judge_model) for judge_model in self.evaluation.judge_models],
        )
        identity = RunIdentity(
            dataset_refs=[
                DatasetRef(
                    dataset_id=dataset.dataset_id,
                    revision=dataset.revision,
                    fingerprint=dataset.compute_hash(),
                )
                for dataset in self.datasets
            ],
            generator_ref=component_refs.generator,
            reducer_ref=component_refs.reducer,
            parser_refs=component_refs.parsers,
            metric_refs=component_refs.metrics,
            judge_model_refs=component_refs.judge_models,
            candidate_policy=self.generation.candidate_policy,
            judge_config=self.evaluation.judge_config,
            workflow_overrides=self.evaluation.workflow_overrides,
            seeds=self.seeds,
        )
        provenance = RunProvenance(
            themis_version=self.themis_version,
            python_version=self.python_version,
            platform=self.platform,
            storage=self.storage,
            runtime=runtime,
            environment_metadata=self.environment_metadata,
        )
        return RunSnapshot(
            identity=identity,
            provenance=provenance,
            component_refs=component_refs,
            datasets=self.datasets,
            metric_kinds=[self._metric_kind(metric) for metric in self.evaluation.metrics],
        )

    async def run_async(
        self,
        *,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        effective_runtime = runtime or self.runtime
        snapshot = self._compile_with_runtime(effective_runtime)
        run_store = store or self._build_store()
        run_store.initialize()
        run_store.persist_snapshot(snapshot)
        orchestrator = Orchestrator(
            store=run_store,
            generator=resolve_generator_component(self.generation.generator),
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parser=resolve_parser_component(self.evaluation.parsers[0]) if self.evaluation.parsers else None,
            metrics=[resolve_metric_component(metric) for metric in self.evaluation.metrics],
            judge_models=[resolve_judge_model_component(judge_model) for judge_model in self.evaluation.judge_models],
            subscribers=subscribers or [],
            tracing_provider=tracing_provider or NoOpTracingProvider(),
            runtime=effective_runtime,
        )
        return await orchestrator.run(snapshot)

    async def rejudge_async(
        self,
        *,
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        effective_runtime = runtime or self.runtime
        snapshot = self._compile_with_runtime(effective_runtime)
        if store is None and self.storage.store == "memory":
            raise ValueError(
                "Memory-backed rejudge requires the original store instance; pass store=... or use sqlite storage."
            )
        run_store = store or self._build_store()
        run_store.initialize()
        if run_store.resume(snapshot.run_id) is None:
            raise ValueError(f"No stored run found for rejudge: {snapshot.run_id}")

        requested_metric_ids = set(metric_ids or [])
        if not requested_metric_ids:
            requested_metric_ids = {
                component_ref.component_id
                for component_ref, metric_kind in zip(snapshot.component_refs.metrics, snapshot.metric_kinds, strict=False)
                if metric_kind != "pure"
            }

        orchestrator = Orchestrator(
            store=run_store,
            generator=resolve_generator_component(self.generation.generator),
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parser=resolve_parser_component(self.evaluation.parsers[0]) if self.evaluation.parsers else None,
            metrics=[resolve_metric_component(metric) for metric in self.evaluation.metrics],
            judge_models=[resolve_judge_model_component(judge_model) for judge_model in self.evaluation.judge_models],
            subscribers=subscribers or [],
            tracing_provider=tracing_provider or NoOpTracingProvider(),
            runtime=effective_runtime,
            force_workflow_metrics=requested_metric_ids,
        )
        return await orchestrator.run(snapshot)

    def run(
        self,
        *,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        return asyncio.run(
            self.run_async(
                runtime=runtime,
                store=store,
                subscribers=subscribers,
                tracing_provider=tracing_provider,
            )
        )

    def rejudge(
        self,
        *,
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        return asyncio.run(
            self.rejudge_async(
                metric_ids=metric_ids,
                runtime=runtime,
                store=store,
                subscribers=subscribers,
                tracing_provider=tracing_provider,
            )
        )

    def _metric_kind(self, metric: object) -> str:
        metric_family = getattr(metric, "metric_family", None)
        if metric_family in {"pure", "llm", "selection", "trace"}:
            return metric_family
        if isinstance(metric, str):
            if metric == "metric/demo":
                return "pure"
            raise ValueError(f"Unknown builtin component: {metric}")
        if isinstance(metric, PureMetric):
            return "pure"
        if isinstance(metric, LLMMetric):
            return "llm"
        if isinstance(metric, SelectionMetric):
            return "selection"
        if isinstance(metric, TraceMetric):
            return "trace"
        raise TypeError("Metrics must satisfy a supported metric protocol.")

    def _build_store(self):
        if self.storage.store == "memory":
            return InMemoryRunStore()
        if self.storage.store == "sqlite":
            path = self.storage.parameters.get("path", "runs/themis.sqlite3")
            return sqlite_store(path)
        raise ValueError(f"Unsupported store backend: {self.storage.store}")
