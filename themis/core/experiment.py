"""Experiment authoring model and snapshot compilation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version as distribution_version
import platform as platform_module
import subprocess
from pathlib import Path
import tomllib
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

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
from themis.core.components import (
    ComponentRef,
    MetricRef,
    component_ref_from_value,
    metric_ref_from_value,
)
from themis.core.config import (
    ExistingRunPolicy,
    EvaluationConfig,
    RuntimeConfig,
    GenerationConfig,
    StorageConfig,
)
from themis.core.dataset_sources import (
    DatasetSourceSpec,
    dataset_materialization_receipt,
    resolved_source_fingerprint,
    resolved_source_id,
    resolved_source_revision,
    inline_dataset_source,
    materialize_dataset_sources,
)
from themis.core.models import Dataset, SeedCapability
from themis.core.orchestrator import Orchestrator
from themis.core.projections import build_run_result, build_run_result_from_state
from themis.core.protocols import (
    EventSubscriber,
    PureMetric,
    WorkflowMetric,
    TracingProvider,
)
from themis.core.registry import RunLineage
from themis.core.results import ExecutionState, RerunPlan, RerunSelector
from themis.core.store import RunStore
from themis.core.stores.factory import create_run_store
from themis.core.tracing import NoOpTracingProvider
from themis.core.snapshot import (
    ComponentRefs,
    DatasetSourceRef,
    ParserViewRef,
    RunIdentity,
    RunProvenance,
    RunSnapshot,
    dataset_manifest_from_dataset,
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


def _raise_if_running_loop(message: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(message)


def _workflow_subject_kind(metric: object) -> str:
    subject_kind = getattr(metric, "subject_kind", None)
    if subject_kind not in {"candidate", "candidates", "trace"}:
        raise ValueError(
            "Workflow metrics must declare subject_kind as candidate, candidates, or trace"
        )
    return str(subject_kind)


def _validate_unique_component_refs(
    label: str,
    refs: list[ComponentRef] | list[MetricRef],
    *,
    allow_identical: bool = False,
) -> None:
    seen: dict[str, ComponentRef | MetricRef] = {}
    for ref in refs:
        if ref.component_id in seen:
            if allow_identical and seen[ref.component_id] == ref:
                continue
            raise ValueError(f"Duplicate {label} component_id: {ref.component_id}")
        seen[ref.component_id] = ref


def _validate_dataset_identifiers(datasets: list[Dataset]) -> None:
    dataset_ids: set[str] = set()
    for dataset in datasets:
        if dataset.dataset_id in dataset_ids:
            raise ValueError(f"Duplicate dataset_id: {dataset.dataset_id}")
        dataset_ids.add(dataset.dataset_id)
        case_ids: set[str] = set()
        for case in dataset.cases:
            if case.case_id in case_ids:
                raise ValueError(
                    f"Duplicate case_id in dataset {dataset.dataset_id}: {case.case_id}"
                )
            case_ids.add(case.case_id)


def _validate_seed_capabilities(experiment: Experiment, runtime: RuntimeConfig) -> None:
    if not runtime.strict_determinism:
        return
    components = [
        resolve_generator_component(experiment.generation.generator),
        *[
            resolve_judge_model_component(model)
            for model in experiment.evaluation.judge_models
        ],
    ]
    unsupported = [
        str(getattr(component, "component_id", component.__class__.__name__))
        for component in components
        if getattr(component, "seed_capability", SeedCapability.UNSUPPORTED)
        != SeedCapability.SUPPORTED
    ]
    if unsupported:
        raise ValueError(
            "Strict determinism requires seed-capable providers; unsupported: "
            + ", ".join(unsupported)
        )


class Experiment(FrozenModel):
    """Authoring model for a Themis experiment.

    An experiment owns the compile-time inputs required to build a `RunSnapshot`
    and provides sync and async helpers for running or rejudging that snapshot.
    """

    generation: GenerationConfig
    evaluation: EvaluationConfig
    storage: StorageConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dataset_sources: Sequence[DatasetSourceSpec | Dataset] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    environment_metadata: dict[str, str] = Field(default_factory=dict)
    themis_version: str = Field(default_factory=_resolve_themis_version)
    python_version: str = Field(default_factory=platform_module.python_version)
    platform: str = Field(default_factory=platform_module.platform)
    git_commit: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    provider_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    _compiled_snapshot: RunSnapshot | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_experiment_payload(cls, payload):
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        sources = normalized.get("dataset_sources")
        if not isinstance(sources, list):
            return normalized
        normalized["dataset_sources"] = [
            item
            if isinstance(item, DatasetSourceSpec)
            else inline_dataset_source(item)
            if isinstance(item, Dataset)
            else item
            if isinstance(item, dict) and "target" in item
            else inline_dataset_source(Dataset.model_validate(item))
            for item in sources
        ]
        return normalized

    def compile(self) -> RunSnapshot:
        """Compile the experiment into an immutable `RunSnapshot`."""

        if self._compiled_snapshot is None:
            self._compiled_snapshot = self._compile_with_runtime(self.runtime)
        return self._compiled_snapshot

    def _compile_with_runtime(self, runtime: RuntimeConfig) -> RunSnapshot:
        dataset_sources = self._resolved_dataset_sources()
        datasets = self._materialized_datasets()
        _validate_dataset_identifiers(datasets)
        _validate_seed_capabilities(self, runtime)
        component_refs = ComponentRefs(
            generator=component_ref_from_value(self.generation.generator),
            selector=component_ref_from_value(self.generation.selector)
            if self.generation.selector is not None
            else None,
            reducer=component_ref_from_value(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=[
                ParserViewRef(
                    id=view.id,
                    parser=component_ref_from_value(view.parser),
                    fallbacks=[
                        component_ref_from_value(fallback)
                        for fallback in view.fallbacks
                    ],
                )
                for view in self.evaluation.parser_views
            ],
            metrics=[
                metric_ref_from_value(metric) for metric in self.evaluation.metrics
            ],
            judge_models=[
                component_ref_from_value(judge_model)
                for judge_model in self.evaluation.judge_models
            ],
        )
        _validate_unique_component_refs("metric", component_refs.metrics)
        _validate_unique_component_refs(
            "judge model", component_refs.judge_models, allow_identical=True
        )
        identity = RunIdentity(
            dataset_source_refs=[
                DatasetSourceRef(
                    dataset_id=dataset.dataset_id,
                    revision=dataset.revision,
                    target=source.target,
                    fingerprint=dataset.compute_hash(),
                    source_id=resolved_source_id(source),
                    source_revision=resolved_source_revision(source),
                    source_fingerprint=resolved_source_fingerprint(source),
                    transform_kwargs=source.transform_kwargs,
                    provenance_metadata=source.provenance_metadata,
                )
                for source, dataset in zip(dataset_sources, datasets, strict=False)
            ],
            generator_ref=component_refs.generator,
            selector_ref=component_refs.selector,
            reducer_ref=component_refs.reducer,
            parser_refs=component_refs.parsers,
            metric_refs=component_refs.metrics,
            judge_model_refs=component_refs.judge_models,
            candidate_policy={
                **self.generation.candidate_policy,
                "max_turns": self.generation.max_turns,
                "termination": self.generation.termination,
            },
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
            dataset_sources=dataset_sources,
            dataset_manifests=[
                dataset_manifest_from_dataset(dataset).model_copy(
                    update={
                        "source_id": resolved_source_id(source),
                        "source_revision": resolved_source_revision(source),
                        "source_fingerprint": resolved_source_fingerprint(source),
                        "materialization_receipt": dataset_materialization_receipt(
                            source, dataset
                        ),
                    }
                )
                for source, dataset in zip(dataset_sources, datasets, strict=False)
            ],
            datasets=datasets,
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
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Run the compiled snapshot asynchronously."""

        effective_runtime = runtime or self.runtime
        snapshot = self._snapshot_for_runtime(effective_runtime)
        run_store = store or self._build_store()
        run_store.initialize()
        resolved_component_refs = self._resolved_component_refs()
        self._validate_component_refs(snapshot, resolved_component_refs)
        existing_state = _fresh_execution_state(run_store, snapshot.run_id)
        stored_run = None
        if existing_state is not None:
            existing_run_policy = effective_runtime.existing_run_policy
            if existing_run_policy is ExistingRunPolicy.ERROR:
                raise ValueError(f"Run already exists for run_id={snapshot.run_id}")
            if existing_run_policy is ExistingRunPolicy.RESTART:
                run_store.clear_run(snapshot.run_id)
                existing_state = None
            elif (
                existing_run_policy is ExistingRunPolicy.REUSE
                and existing_state.status.value == "completed"
                and _stage_index(existing_state.completed_through_stage)
                >= _stage_index(until_stage)
            ):
                return build_run_result_from_state(snapshot, existing_state)
        else:
            stored_run = run_store.resume(snapshot.run_id)
            if stored_run is not None:
                existing_run_policy = effective_runtime.existing_run_policy
                if existing_run_policy is ExistingRunPolicy.ERROR:
                    raise ValueError(f"Run already exists for run_id={snapshot.run_id}")
                if existing_run_policy is ExistingRunPolicy.RESTART:
                    run_store.clear_run(snapshot.run_id)
                    stored_run = None
                elif (
                    existing_run_policy is ExistingRunPolicy.REUSE
                    and stored_run.execution_state.status.value == "completed"
                    and _stage_index(stored_run.execution_state.completed_through_stage)
                    >= _stage_index(until_stage)
                ):
                    return build_run_result(stored_run.snapshot, stored_run.events)
        if existing_state is None and stored_run is None:
            run_store.persist_snapshot(snapshot)
        orchestrator = Orchestrator(
            store=run_store,
            generator=resolve_generator_component(self.generation.generator),
            selector=resolve_selector_component(self.generation.selector)
            if self.generation.selector is not None
            else None,
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=self._resolved_parser_views(),
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
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
        _attempt_kind: Literal["replay", "rejudge"] = "replay",
    ):
        """Replay persisted runs from a downstream stage."""

        if stage not in {"reduce", "parse", "score", "judge"}:
            raise ValueError(f"Unsupported replay stage: {stage}")

        effective_runtime = runtime or self.runtime
        snapshot = self._snapshot_for_runtime(effective_runtime)
        if store is None and self.storage.target == "memory":
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
            parsers=self._resolved_parser_views(),
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
            attempt_kind=_attempt_kind,
        )
        return await orchestrator.run(snapshot)

    async def rejudge_async(
        self,
        *,
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[EventSubscriber] | None = None,
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
            _attempt_kind="rejudge",
        )

    async def rerun_async(
        self,
        *,
        stage: Literal["generate", "reduce", "parse", "score", "judge"],
        failed_only: bool = False,
        case_ids: list[str] | None = None,
        case_keys: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Rerun a targeted subset of an existing stored run."""

        effective_runtime = runtime or self.runtime
        snapshot = self._snapshot_for_runtime(effective_runtime)
        if store is None and self.storage.target == "memory":
            raise ValueError(
                "Memory-backed rerun requires the original store instance; pass store=... or use sqlite storage."
            )
        run_store = store or self._build_store()
        run_store.initialize()
        if run_store.load_execution_checkpoint(snapshot.run_id) is None and (
            run_store.resume(snapshot.run_id) is None
        ):
            raise ValueError(f"No stored run found for rerun: {snapshot.run_id}")
        self._validate_component_refs(snapshot, self._resolved_component_refs())
        rerun_plan = RerunPlan(
            stage=stage,
            selector=RerunSelector(
                failed_only=failed_only,
                case_ids=list(case_ids or []),
                case_keys=list(case_keys or []),
                metadata=dict(metadata or {}),
                metric_ids=list(metric_ids or []),
            ),
        )
        orchestrator = Orchestrator(
            store=run_store,
            generator=resolve_generator_component(self.generation.generator),
            selector=resolve_selector_component(self.generation.selector)
            if self.generation.selector is not None
            else None,
            reducer=resolve_reducer_component(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=self._resolved_parser_views(),
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
            force_workflow_metrics=set(metric_ids or []) if stage == "judge" else set(),
            rerun_plan=rerun_plan,
            attempt_kind="rerun",
        )
        parent_attempt_id = next(
            (
                event.attempt_id
                for event in reversed(run_store.query_events(snapshot.run_id))
                if event.attempt_id != orchestrator.runtime_support.attempt_id
            ),
            None,
        )
        result = await orchestrator.run(snapshot)
        record = run_store.get_run_record(snapshot.run_id)
        lineage = list(record.lineage) if record is not None else []
        lineage.append(
            RunLineage(
                parent_run_id=snapshot.run_id,
                parent_attempt_id=parent_attempt_id,
                relationship="rerun",
            )
        )
        run_store.update_run_record(snapshot.run_id, lineage=lineage)
        return result

    def run(
        self,
        *,
        until_stage: Literal["generate", "reduce", "parse", "score", "judge"] = "judge",
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Run the compiled snapshot synchronously."""

        _raise_if_running_loop(
            "Experiment.run() cannot be called from a running event loop. Use await Experiment.run_async()."
        )
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
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Re-run workflow-backed metrics synchronously."""

        _raise_if_running_loop(
            "Experiment.rejudge() cannot be called from a running event loop. Use await Experiment.rejudge_async()."
        )
        return asyncio.run(
            self.rejudge_async(
                metric_ids=metric_ids,
                runtime=runtime,
                store=store,
                subscribers=subscribers,
                tracing_provider=tracing_provider,
            )
        )

    def rerun(
        self,
        *,
        stage: Literal["generate", "reduce", "parse", "score", "judge"],
        failed_only: bool = False,
        case_ids: list[str] | None = None,
        case_keys: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        metric_ids: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
        store: RunStore | None = None,
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Rerun a targeted subset of an existing stored run synchronously."""

        _raise_if_running_loop(
            "Experiment.rerun() cannot be called from a running event loop. Use await Experiment.rerun_async()."
        )
        return asyncio.run(
            self.rerun_async(
                stage=stage,
                failed_only=failed_only,
                case_ids=case_ids,
                case_keys=case_keys,
                metadata=metadata,
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
        subscribers: list[EventSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
    ):
        """Replay persisted runs from a downstream stage synchronously."""

        _raise_if_running_loop(
            "Experiment.replay() cannot be called from a running event loop. Use await Experiment.replay_async()."
        )
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

    def _resolved_dataset_sources(self) -> list[DatasetSourceSpec]:
        return [
            source
            if isinstance(source, DatasetSourceSpec)
            else inline_dataset_source(source)
            for source in self.dataset_sources
        ]

    def _materialized_datasets(self) -> list[Dataset]:
        return materialize_dataset_sources(self._resolved_dataset_sources())

    @property
    def datasets(self) -> list[Dataset]:
        """Return the materialized runtime datasets for this experiment."""

        return self._materialized_datasets()

    def _metric_kind(self, metric: object) -> str:
        metric_family = getattr(metric, "metric_family", None)
        if metric_family == "pure":
            return "pure"
        if metric_family == "workflow":
            return _workflow_subject_kind(metric)
        target = getattr(metric, "target", None)
        kwargs = getattr(metric, "kwargs", None)
        if isinstance(target, str) and isinstance(kwargs, dict):
            resolved_metric = resolve_metric_component(metric)
            resolved_metric_family = getattr(resolved_metric, "metric_family", None)
            if resolved_metric_family == "pure":
                return "pure"
            if resolved_metric_family == "workflow":
                return _workflow_subject_kind(resolved_metric)
            raise ValueError(f"Unknown metric family for target spec: {target}")
        if isinstance(metric, str):
            resolved_metric = resolve_metric_component(metric)
            resolved_metric_family = getattr(resolved_metric, "metric_family", None)
            if resolved_metric_family == "pure":
                return "pure"
            if resolved_metric_family == "workflow":
                return _workflow_subject_kind(resolved_metric)
            raise ValueError(f"Unknown builtin metric family: {metric}")
        if isinstance(metric, PureMetric):
            return "pure"
        if isinstance(metric, WorkflowMetric):
            return _workflow_subject_kind(metric)
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
                ParserViewRef(
                    id=view.id,
                    parser=component_ref_from_value(
                        resolve_parser_component(view.parser)
                    ),
                    fallbacks=[
                        component_ref_from_value(resolve_parser_component(fallback))
                        for fallback in view.fallbacks
                    ],
                )
                for view in self.evaluation.parser_views
            ],
            metrics=[
                metric_ref_from_value(resolve_metric_component(metric))
                for metric in self.evaluation.metrics
            ],
            judge_models=[
                component_ref_from_value(resolve_judge_model_component(judge_model))
                for judge_model in self.evaluation.judge_models
            ],
        )

    def _resolved_parser_views(self):
        return [
            (
                view.id,
                resolve_parser_component(view.parser),
                [resolve_parser_component(fallback) for fallback in view.fallbacks],
            )
            for view in self.evaluation.parser_views
        ]

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
        self._validate_parser_ref_list(
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
        expected: list[ComponentRef] | list[MetricRef],
        actual: list[ComponentRef] | list[MetricRef],
    ) -> None:
        if expected == actual:
            return
        raise RuntimeError(
            f"Component fingerprint mismatch for {label}: expected {expected}, got {actual}. Recompile the experiment."
        )

    def _validate_parser_ref_list(
        self,
        label: str,
        expected: list[ParserViewRef],
        actual: list[ParserViewRef],
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


def _fresh_execution_state(store: RunStore, run_id: str) -> ExecutionState | None:
    checkpoint = store.load_execution_checkpoint(run_id)
    if checkpoint is None:
        return None
    if checkpoint.event_count != store.count_events(run_id):
        return None
    return checkpoint.execution_state


def _default_dependency_versions(themis_version: str) -> dict[str, str]:
    versions = {"themis-eval": themis_version}
    for distribution in ("pydantic", "omegaconf", "cyclopts"):
        try:
            versions[distribution] = distribution_version(distribution)
        except PackageNotFoundError:
            continue
    return versions


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
