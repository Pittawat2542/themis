"""Experiment authoring model and snapshot compilation for Phase 1."""

from __future__ import annotations

import asyncio

from pydantic import Field

from themis.core.builtins import (
    resolve_generator_component,
    resolve_metric_component,
    resolve_parser_component,
    resolve_reducer_component,
)
from themis.core.base import FrozenModel
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Dataset
from themis.core.orchestrator import Orchestrator
from themis.core.protocols import LLMMetric, PureMetric, SelectionMetric, TraceMetric
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.sqlite import sqlite_store
from themis.core.tracing import NoOpTracingProvider
from themis.core.snapshot import (
    ComponentRefs,
    DatasetRef,
    RunIdentity,
    RunProvenance,
    RunSnapshot,
    component_ref_from_value,
)


class Experiment(FrozenModel):
    """Compiled-input experiment definition for Themis v4."""

    generation: GenerationConfig
    evaluation: EvaluationConfig
    storage: StorageConfig
    datasets: list[Dataset] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    environment_metadata: dict[str, str] = Field(default_factory=dict)
    themis_version: str = "4.0.0a0"
    python_version: str = "3.12"
    platform: str = "unknown"

    def compile(self) -> RunSnapshot:
        component_refs = ComponentRefs(
            generator=component_ref_from_value(self.generation.generator),
            reducer=component_ref_from_value(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=[component_ref_from_value(parser) for parser in self.evaluation.parsers],
            metrics=[component_ref_from_value(metric) for metric in self.evaluation.metrics],
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
            environment_metadata=self.environment_metadata,
        )
        return RunSnapshot(
            identity=identity,
            provenance=provenance,
            component_refs=component_refs,
            datasets=self.datasets,
            metric_kinds=[self._metric_kind(metric) for metric in self.evaluation.metrics],
        )

    async def run_async(self, *, subscribers: list[object] | None = None, tracing_provider=None):
        snapshot = self.compile()
        store = self._build_store()
        store.initialize()
        store.persist_snapshot(snapshot)
        orchestrator = Orchestrator(
            store=store,
            generator=resolve_generator_component(self.generation.generator),
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parser=resolve_parser_component(self.evaluation.parsers[0]) if self.evaluation.parsers else None,
            metrics=[resolve_metric_component(metric) for metric in self.evaluation.metrics],
            subscribers=subscribers or [],
            tracing_provider=tracing_provider or NoOpTracingProvider(),
        )
        return await orchestrator.run(snapshot)

    def run(self, *, subscribers: list[object] | None = None, tracing_provider=None):
        return asyncio.run(
            self.run_async(subscribers=subscribers, tracing_provider=tracing_provider)
        )

    def _metric_kind(self, metric: object) -> str:
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
