"""Experiment authoring model and snapshot compilation."""

from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version as distribution_version
import subprocess
from pathlib import Path
import tomllib
from typing import Literal

from pydantic import Field, PrivateAttr

from themis.core.builtins import (
    resolve_generator_component,
    resolve_judge_model_component,
    resolve_metric_component,
    resolve_parser_component,
    resolve_reducer_component,
    resolve_selector_component,
)
from themis.core.base import FrozenModel
from themis.core.base import JSONValue
from themis.core.components import ComponentRef, component_ref_from_value
from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    RuntimeConfig,
    StorageConfig,
)
from themis.core.config_loading import ExperimentConfigMetadata
from themis.core.models import Dataset
from themis.core.orchestrator import Orchestrator
from themis.core.projections import build_run_result
from themis.core.protocols import (
    LLMMetric,
    LifecycleSubscriber,
    PureMetric,
    SelectionMetric,
    TraceMetric,
    TracingProvider,
)
from themis.core.store import RunStore
from themis.core.stores.factory import create_run_store
from themis.core.tracing import NoOpTracingProvider
from themis.core.snapshot import (
    ComponentRefs,
    DatasetRef,
    RunIdentity,
    RunProvenance,
    RunSnapshot,
)


def _resolve_themis_version() -> str:
    """Resolve the release version for runtime provenance defaults."""

    try:
        return distribution_version("themis-eval")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject_path.is_file():
            payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            return str(payload["project"]["version"])
        return "0+unknown"


class Experiment(FrozenModel):
    """Authoring model for a Themis experiment.

    An experiment owns the compile-time inputs required to build a `RunSnapshot`
    and provides sync and async helpers for running or rejudging that snapshot.
    """

    generation: GenerationConfig
    evaluation: EvaluationConfig
    storage: StorageConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    datasets: list[Dataset] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    environment_metadata: dict[str, str] = Field(default_factory=dict)
    themis_version: str = Field(default_factory=_resolve_themis_version)
    python_version: str = "3.12"
    platform: str = "unknown"
    git_commit: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    provider_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    _config_metadata: ExperimentConfigMetadata | None = PrivateAttr(default=None)
    _compiled_snapshot: RunSnapshot | None = PrivateAttr(default=None)

    @classmethod
    def from_config(
        cls, path: str | Path, *, overrides: list[str] | None = None
    ) -> Experiment:
        """Load an experiment definition from YAML or TOML configuration."""

        from themis.core.config_loading import load_experiment_definition

        loaded = load_experiment_definition(path, overrides=overrides)
        experiment = cls.model_validate(loaded.payload)
        experiment._config_metadata = loaded.metadata
        return experiment

    def compile(self) -> RunSnapshot:
        """Compile the experiment into an immutable `RunSnapshot`."""

        if self._compiled_snapshot is None:
            self._compiled_snapshot = self._compile_with_runtime(self.runtime)
        return self._compiled_snapshot

    def _compile_with_runtime(self, runtime: RuntimeConfig) -> RunSnapshot:
        component_refs = ComponentRefs(
            generator=component_ref_from_value(self.generation.generator),
            selector=component_ref_from_value(self.generation.selector)
            if self.generation.selector is not None
            else None,
            reducer=component_ref_from_value(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=[
                component_ref_from_value(parser) for parser in self.evaluation.parsers
            ],
            metrics=[
                component_ref_from_value(metric) for metric in self.evaluation.metrics
            ],
            judge_models=[
                component_ref_from_value(judge_model)
                for judge_model in self.evaluation.judge_models
            ],
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
            selector_ref=component_refs.selector,
            reducer_ref=component_refs.reducer,
            parser_refs=component_refs.parsers,
            metric_refs=component_refs.metrics,
            judge_model_refs=component_refs.judge_models,
            candidate_policy=self.generation.candidate_policy,
            generation_prompt_spec=self.generation.prompt_spec,
            evaluation_prompt_spec=self.evaluation.prompt_spec,
            judge_config=self.evaluation.judge_config,
            workflow_overrides=self.evaluation.workflow_overrides,
            seeds=self.seeds,
        ).sanitized()
        provenance = RunProvenance(
            themis_version=self.themis_version,
            python_version=self.python_version,
            platform=self.platform,
            git_commit=self.git_commit or _detect_git_commit(),
            dependency_versions=self.dependency_versions
            or _default_dependency_versions(self.themis_version),
            provider_metadata=self.provider_metadata
            or _default_provider_metadata(self),
            storage=self.storage,
            runtime=runtime,
            environment_metadata=self.environment_metadata,
        ).sanitized()
        return RunSnapshot(
            identity=identity,
            provenance=provenance,
            component_refs=component_refs,
            datasets=self.datasets,
            metric_kinds=[
                self._metric_kind(metric) for metric in self.evaluation.metrics
            ],
        )

    async def run_async(
        self,
        *,
        until_stage: Literal["generate", "reduce", "parse", "score", "judge"] = "judge",
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Run the compiled snapshot asynchronously."""

        effective_runtime = runtime or self.runtime
        snapshot = self._snapshot_for_runtime(effective_runtime)
        run_store = store or self._build_store()
        run_store.initialize()
        stored_run = run_store.resume(snapshot.run_id)
        if stored_run is not None:
            existing_run_policy = effective_runtime.existing_run_policy
            if existing_run_policy == "error":
                raise ValueError(f"Run already exists for run_id={snapshot.run_id}")
            if existing_run_policy == "rerun":
                run_store.clear_run(snapshot.run_id)
                stored_run = None
            elif (
                existing_run_policy == "auto"
                and stored_run.execution_state.status.value == "completed"
                and _stage_index(stored_run.execution_state.completed_through_stage)
                >= _stage_index(until_stage)
            ):
                return build_run_result(stored_run.snapshot, stored_run.events)
        if stored_run is None:
            run_store.persist_snapshot(snapshot)
        resolved_component_refs = self._resolved_component_refs()
        self._validate_component_refs(snapshot, resolved_component_refs)
        orchestrator = Orchestrator(
            store=run_store,
            generator=resolve_generator_component(self.generation.generator),
            selector=resolve_selector_component(self.generation.selector)
            if self.generation.selector is not None
            else None,
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parser=resolve_parser_component(self.evaluation.parsers[0])
            if self.evaluation.parsers
            else None,
            metrics=[
                resolve_metric_component(metric) for metric in self.evaluation.metrics
            ],
            judge_models=[
                resolve_judge_model_component(judge_model)
                for judge_model in self.evaluation.judge_models
            ],
            subscribers=subscribers or [],
            tracing_provider=tracing_provider or NoOpTracingProvider(),
            runtime=effective_runtime,
            until_stage=until_stage,
        )
        return await orchestrator.run(snapshot)

    async def replay_async(
        self,
        *,
        stage: Literal["reduce", "parse", "score", "judge"],
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Replay persisted runs from a downstream stage."""

        if stage not in {"reduce", "parse", "score", "judge"}:
            raise ValueError(f"Unsupported replay stage: {stage}")

        effective_runtime = runtime or self.runtime
        snapshot = self._snapshot_for_runtime(effective_runtime)
        if store is None and self.storage.store == "memory":
            raise ValueError(
                "Memory-backed replay requires the original store instance; pass store=... or use sqlite storage."
            )
        run_store = store or self._build_store()
        run_store.initialize()
        if run_store.resume(snapshot.run_id) is None:
            raise ValueError(f"No stored run found for replay: {snapshot.run_id}")
        self._validate_component_refs(snapshot, self._resolved_component_refs())

        requested_metric_ids = set(metric_ids or [])
        if stage == "judge" and not requested_metric_ids:
            requested_metric_ids = {
                component_ref.component_id
                for component_ref, metric_kind in zip(
                    snapshot.component_refs.metrics, snapshot.metric_kinds, strict=False
                )
                if metric_kind != "pure"
            }

        orchestrator = Orchestrator(
            store=run_store,
            generator=resolve_generator_component(self.generation.generator),
            selector=resolve_selector_component(self.generation.selector)
            if self.generation.selector is not None
            else None,
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parser=resolve_parser_component(self.evaluation.parsers[0])
            if self.evaluation.parsers
            else None,
            metrics=[
                resolve_metric_component(metric) for metric in self.evaluation.metrics
            ],
            judge_models=[
                resolve_judge_model_component(judge_model)
                for judge_model in self.evaluation.judge_models
            ],
            subscribers=subscribers or [],
            tracing_provider=tracing_provider or NoOpTracingProvider(),
            runtime=effective_runtime,
            force_workflow_metrics=requested_metric_ids,
            replay_stage=stage,
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
        """Re-run workflow-backed metrics from stored upstream artifacts."""
        return await self.replay_async(
            stage="judge",
            metric_ids=metric_ids,
            runtime=runtime,
            store=store,
            subscribers=subscribers,
            tracing_provider=tracing_provider,
        )

    def run(
        self,
        *,
        until_stage: Literal["generate", "reduce", "parse", "score", "judge"] = "judge",
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Run the compiled snapshot synchronously."""

        return asyncio.run(
            self.run_async(
                until_stage=until_stage,
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
        """Re-run workflow-backed metrics synchronously."""

        return asyncio.run(
            self.rejudge_async(
                metric_ids=metric_ids,
                runtime=runtime,
                store=store,
                subscribers=subscribers,
                tracing_provider=tracing_provider,
            )
        )

    def replay(
        self,
        *,
        stage: Literal["reduce", "parse", "score", "judge"],
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Replay persisted runs from a downstream stage synchronously."""

        return asyncio.run(
            self.replay_async(
                stage=stage,
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
            resolved_metric = resolve_metric_component(metric)
            resolved_metric_family = getattr(resolved_metric, "metric_family", None)
            if resolved_metric_family in {"pure", "llm", "selection", "trace"}:
                return resolved_metric_family
            raise ValueError(f"Unknown builtin metric family: {metric}")
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
        return create_run_store(self.storage)

    def _snapshot_for_runtime(self, runtime: RuntimeConfig) -> RunSnapshot:
        compiled = self.compile()
        if runtime == compiled.provenance.runtime:
            return compiled
        return compiled.model_copy(
            update={
                "provenance": compiled.provenance.model_copy(
                    update={"runtime": runtime}
                ),
            }
        )

    def _resolved_component_refs(self) -> ComponentRefs:
        return ComponentRefs(
            generator=component_ref_from_value(
                resolve_generator_component(self.generation.generator)
            ),
            selector=component_ref_from_value(
                resolve_selector_component(self.generation.selector)
            )
            if self.generation.selector is not None
            else None,
            reducer=component_ref_from_value(
                resolve_reducer_component(self.generation.reducer)
            )
            if self.generation.reducer is not None
            else None,
            parsers=[
                component_ref_from_value(resolve_parser_component(parser))
                for parser in self.evaluation.parsers
            ],
            metrics=[
                component_ref_from_value(resolve_metric_component(metric))
                for metric in self.evaluation.metrics
            ],
            judge_models=[
                component_ref_from_value(resolve_judge_model_component(judge_model))
                for judge_model in self.evaluation.judge_models
            ],
        )

    def _validate_component_refs(
        self, snapshot: RunSnapshot, resolved: ComponentRefs
    ) -> None:
        self._validate_component_ref(
            "generator", snapshot.component_refs.generator, resolved.generator
        )
        self._validate_component_ref(
            "selector", snapshot.component_refs.selector, resolved.selector
        )
        self._validate_component_ref(
            "reducer", snapshot.component_refs.reducer, resolved.reducer
        )
        self._validate_component_ref_list(
            "parser", snapshot.component_refs.parsers, resolved.parsers
        )
        self._validate_component_ref_list(
            "metric", snapshot.component_refs.metrics, resolved.metrics
        )
        self._validate_component_ref_list(
            "judge_model", snapshot.component_refs.judge_models, resolved.judge_models
        )

    def _validate_component_ref(
        self,
        label: str,
        expected: ComponentRef | None,
        actual: ComponentRef | None,
    ) -> None:
        if expected == actual:
            return
        raise RuntimeError(
            f"Component fingerprint mismatch for {label}: expected {expected}, got {actual}. Recompile the experiment."
        )

    def _validate_component_ref_list(
        self,
        label: str,
        expected: list[ComponentRef],
        actual: list[ComponentRef],
    ) -> None:
        if expected == actual:
            return
        raise RuntimeError(
            f"Component fingerprint mismatch for {label}: expected {expected}, got {actual}. Recompile the experiment."
        )


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _stage_index(stage: str | None) -> int:
    order = {
        "generate": 0,
        "reduce": 1,
        "parse": 2,
        "score": 3,
        "judge": 4,
    }
    return order.get(stage or "judge", 4)


def _default_dependency_versions(themis_version: str) -> dict[str, str]:
    return {"themis-eval": themis_version}


def _default_provider_metadata(experiment: Experiment) -> dict[str, JSONValue]:
    metadata: dict[str, JSONValue] = {}
    for label, value in (
        ("generator", experiment.generation.generator),
        *[
            (f"judge_model:{index}", judge_model)
            for index, judge_model in enumerate(experiment.evaluation.judge_models)
        ],
    ):
        provider_key = getattr(value, "provider_key", None)
        model_id = getattr(value, "model_id", None)
        endpoint = getattr(value, "endpoint", None)
        entry: dict[str, JSONValue] = {}
        if isinstance(provider_key, str):
            entry["provider_key"] = provider_key
        if isinstance(model_id, str):
            entry["model_id"] = model_id
        if isinstance(endpoint, str):
            entry["endpoint"] = endpoint
        if entry:
            metadata[label] = entry
    return metadata
