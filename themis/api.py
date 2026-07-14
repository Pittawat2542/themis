"""Small public API for authoring and running Themis v6 experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from themis.core.base import FrozenModel, JSONValue
from themis.core.config import (
    EvidenceRetention,
    ExistingRunPolicy,
    GeneratorComponent,
    JudgeModelComponent,
    MetricComponent,
    ParserComponent,
    ParserView,
    ReducerComponent,
    RuntimeConfig,
    SelectorComponent,
    Stage,
    StorageConfig,
    EvaluationConfig,
    GenerationConfig,
)
from themis.core.dataset_sources import DatasetSourceSpec
from themis.core.experiment import Experiment as _CoreExperiment
from themis.core.models import Case, Dataset
from themis.core.prompts import PromptSpec
from themis.core.protocols import EventSubscriber, TracingProvider
from themis.core.results import RunResult
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.tracing import NoOpTracingProvider
from themis.storage import memory_store, storage_config

PositiveInt = Annotated[int, Field(ge=1)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
BackoffFactor = Annotated[float, Field(ge=1.0)]
UntilStage = Literal["generate", "reduce", "parse", "score", "judge"]
ReplayStage = Literal["reduce", "parse", "score", "judge"]


class DatasetSource(DatasetSourceSpec):
    """Identity-bearing source that can materialize a dataset."""


class Generation(FrozenModel):
    """Identity-bearing candidate generation definition."""

    generator: GeneratorComponent
    samples: PositiveInt = 1
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None
    prompt: PromptSpec | None = None
    max_turns: PositiveInt = 1
    termination: dict[str, JSONValue] = Field(default_factory=dict)

    def _core(self) -> GenerationConfig:
        return GenerationConfig(
            generator=self.generator,
            candidate_policy={"num_samples": self.samples},
            prompt_spec=self.prompt,
            max_turns=self.max_turns,
            termination=self.termination,
            selector=self.selector,
            reducer=self.reducer,
        )


class Evaluation(FrozenModel):
    """Identity-bearing parsing, scoring, and judging definition."""

    metrics: list[MetricComponent] = Field(default_factory=list)
    parser: ParserComponent | None = None
    parser_views: Mapping[str, ParserComponent] = Field(default_factory=dict)
    judge_models: list[JudgeModelComponent] = Field(default_factory=list)
    prompt: PromptSpec | None = None
    judge_options: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_options: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_parser_surface(self) -> Evaluation:
        if self.parser is not None and self.parser_views:
            raise ValueError("Use parser or parser_views, not both")
        return self

    def _core(self) -> EvaluationConfig:
        parsers: list[ParserView | ParserComponent]
        if self.parser is not None:
            parsers = [self.parser]
        else:
            parsers = [
                ParserView(id=view_id, parser=parser)
                for view_id, parser in self.parser_views.items()
            ]
        return EvaluationConfig(
            metrics=self.metrics,
            parsers=parsers,
            judge_models=self.judge_models,
            prompt_spec=self.prompt,
            judge_config=self.judge_options,
            workflow_overrides=self.workflow_options,
        )


class RunOptions(FrozenModel):
    """Validated execution controls excluded from logical experiment identity."""

    max_concurrency: PositiveInt = 32
    stage_concurrency: dict[Stage, PositiveInt] = Field(default_factory=dict)
    provider_concurrency: dict[str, PositiveInt] = Field(default_factory=dict)
    provider_rate_limits: dict[str, PositiveInt] = Field(default_factory=dict)
    provider_token_limits: dict[str, PositiveInt] = Field(default_factory=dict)
    provider_timeout_seconds: PositiveFloat = 120.0
    generation_retry_attempts: PositiveInt = 3
    generation_retry_delay: NonNegativeFloat = 0.01
    generation_retry_backoff: BackoffFactor = 2.0
    judge_retry_attempts: PositiveInt = 3
    judge_retry_delay: NonNegativeFloat = 0.01
    judge_retry_backoff: BackoffFactor = 2.0
    store_retry_attempts: PositiveInt = 5
    store_retry_delay: NonNegativeFloat = 0.01
    strict_determinism: bool = False
    evidence_retention: EvidenceRetention = EvidenceRetention.STANDARD
    persistence_timeout_seconds: PositiveFloat = 30.0
    subscriber_timeout_seconds: PositiveFloat = 5.0
    evidence_queue_capacity: PositiveInt = 256
    evidence_batch_size: PositiveInt = 1
    existing_run_policy: ExistingRunPolicy = ExistingRunPolicy.REUSE

    def _core(self) -> RuntimeConfig:
        return RuntimeConfig(
            max_concurrent_tasks=self.max_concurrency,
            stage_concurrency=self.stage_concurrency,
            provider_concurrency=self.provider_concurrency,
            provider_rate_limits=self.provider_rate_limits,
            provider_token_limits=self.provider_token_limits,
            provider_timeout_seconds=self.provider_timeout_seconds,
            generation_retry_attempts=self.generation_retry_attempts,
            generation_retry_delay=self.generation_retry_delay,
            generation_retry_backoff=self.generation_retry_backoff,
            judge_retry_attempts=self.judge_retry_attempts,
            judge_retry_delay=self.judge_retry_delay,
            judge_retry_backoff=self.judge_retry_backoff,
            store_retry_attempts=self.store_retry_attempts,
            store_retry_delay=self.store_retry_delay,
            strict_determinism=self.strict_determinism,
            evidence_retention=self.evidence_retention,
            persistence_timeout_seconds=self.persistence_timeout_seconds,
            subscriber_timeout_seconds=self.subscriber_timeout_seconds,
            evidence_queue_capacity=self.evidence_queue_capacity,
            evidence_batch_size=self.evidence_batch_size,
            existing_run_policy=self.existing_run_policy,
        )


class Experiment(FrozenModel):
    """Canonical, immutable definition of a reproducible evaluation experiment."""

    datasets: Sequence[DatasetSourceSpec | Dataset]
    generation: Generation
    evaluation: Evaluation
    seeds: list[int] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def compile(
        self,
        *,
        store: RunStore | None = None,
        options: RunOptions | None = None,
    ) -> RunSnapshot:
        """Compile identity and the supplied execution provenance."""

        resolved_options = options or RunOptions()
        return self._core(
            storage_config(store) if store is not None else StorageConfig(),
            resolved_options,
        ).compile()

    async def run_async(
        self,
        *,
        store: RunStore,
        options: RunOptions | None = None,
        until_stage: Stage = Stage.JUDGE,
        subscribers: Sequence[EventSubscriber] = (),
        tracing_provider: TracingProvider | None = None,
    ) -> RunResult:
        """Execute this experiment asynchronously with explicit persistence."""

        resolved_options = options or RunOptions()
        core = self._core(storage_config(store), resolved_options)
        return await core.run_async(
            store=store,
            runtime=resolved_options._core(),
            until_stage=cast(UntilStage, until_stage.value),
            subscribers=list(subscribers),
            tracing_provider=tracing_provider or NoOpTracingProvider(),
        )

    def run(
        self,
        *,
        store: RunStore,
        options: RunOptions | None = None,
        until_stage: Stage = Stage.JUDGE,
        subscribers: Sequence[EventSubscriber] = (),
        tracing_provider: TracingProvider | None = None,
    ) -> RunResult:
        """Execute this experiment synchronously with explicit persistence."""

        resolved_options = options or RunOptions()
        core = self._core(storage_config(store), resolved_options)
        return core.run(
            store=store,
            runtime=resolved_options._core(),
            until_stage=cast(UntilStage, until_stage.value),
            subscribers=list(subscribers),
            tracing_provider=tracing_provider or NoOpTracingProvider(),
        )

    def replay(
        self,
        *,
        store: RunStore,
        from_stage: Stage,
        metric_ids: Sequence[str] = (),
        options: RunOptions | None = None,
    ) -> RunResult:
        """Replay persisted evidence from a downstream stage."""

        if from_stage in {Stage.GENERATE, Stage.SELECT}:
            raise ValueError("Replay must start from reduce, parse, score, or judge")
        resolved_options = options or RunOptions()
        return self._core(storage_config(store), resolved_options).replay(
            store=store,
            runtime=resolved_options._core(),
            stage=cast(ReplayStage, from_stage.value),
            metric_ids=list(metric_ids),
        )

    def rerun(
        self,
        *,
        store: RunStore,
        from_stage: Stage,
        case_ids: Sequence[str] = (),
        metric_ids: Sequence[str] = (),
        failed_only: bool = False,
        options: RunOptions | None = None,
    ) -> RunResult:
        """Rerun a selected subset while recording lineage."""

        if from_stage is Stage.SELECT:
            raise ValueError("Selection is not a persisted rerun boundary")
        resolved_options = options or RunOptions()
        return self._core(storage_config(store), resolved_options).rerun(
            store=store,
            runtime=resolved_options._core(),
            stage=cast(UntilStage, from_stage.value),
            case_ids=list(case_ids),
            metric_ids=list(metric_ids),
            failed_only=failed_only,
        )

    def _core(self, storage: StorageConfig, options: RunOptions) -> _CoreExperiment:
        return _CoreExperiment(
            generation=self.generation._core(),
            evaluation=self.evaluation._core(),
            storage=storage,
            runtime=options._core(),
            dataset_sources=list(self.datasets),
            seeds=self.seeds,
            environment_metadata=self.metadata,
        )


def evaluate(
    *,
    cases: Sequence[Case],
    generator: GeneratorComponent,
    metrics: Sequence[MetricComponent],
    parser: ParserComponent | None = None,
    store: RunStore | None = None,
) -> RunResult:
    """Run a deliberately small, in-memory-friendly evaluation."""

    run_store = store or memory_store()
    experiment = Experiment(
        datasets=[Dataset(dataset_id="evaluation", cases=list(cases))],
        generation=Generation(generator=generator),
        evaluation=Evaluation(metrics=list(metrics), parser=parser),
    )
    return experiment.run(store=run_store)
