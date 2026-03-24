"""Resolve concrete plugins and hook adapters for one trial session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from themis.contracts.protocols import (
    Extractor,
    InferenceEngine,
    InferenceResult,
    JudgeService,
    Metric,
    RenderedPrompt,
    TraceMetric,
    TrialMetric,
)
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
    ResolvedTaskStages,
    resolve_task_stages,
)
from themis.records.candidate import CandidateRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import TrialSpec
from themis.specs.foundational import MetricRefSpec
from themis.types.enums import RunStage
from themis.types.json_types import JSONValueType

PreInferenceHook = Callable[[TrialSpec, RenderedPrompt], RenderedPrompt]
PostInferenceHook = Callable[[TrialSpec, InferenceResult], InferenceResult]
CandidateHook = Callable[[TrialSpec, CandidateRecord], CandidateRecord]
JudgeEngineResolver = Callable[[str], InferenceEngine]
ResolvedStage: TypeAlias = RunStage


@dataclass(frozen=True, slots=True)
class ResolvedPipelineHooks:
    """Typed hook callables resolved once from arbitrary registered hook objects."""

    pre_inference: tuple[PreInferenceHook, ...] = ()
    post_inference: tuple[PostInferenceHook, ...] = ()
    pre_extraction: tuple[CandidateHook, ...] = ()
    post_extraction: tuple[CandidateHook, ...] = ()
    pre_eval: tuple[CandidateHook, ...] = ()
    post_eval: tuple[CandidateHook, ...] = ()

    @classmethod
    def from_registry(cls, registry: PluginRegistry) -> ResolvedPipelineHooks:
        registrations = registry.iter_hook_registrations()
        return cls(
            pre_inference=tuple(
                registration.hook.pre_inference for registration in registrations
            ),
            post_inference=tuple(
                registration.hook.post_inference for registration in registrations
            ),
            pre_extraction=tuple(
                registration.hook.pre_extraction for registration in registrations
            ),
            post_extraction=tuple(
                registration.hook.post_extraction for registration in registrations
            ),
            pre_eval=tuple(
                registration.hook.pre_eval for registration in registrations
            ),
            post_eval=tuple(
                registration.hook.post_eval for registration in registrations
            ),
        )

    def apply_pre_inference(self, trial: TrialSpec) -> TrialSpec:
        prompt = RenderedPrompt(
            messages=list(trial.prompt.messages),
            follow_up_turns=list(trial.prompt.follow_up_turns),
            tools=list(trial.tools),
            mcp_servers=list(trial.mcp_servers),
        )
        for hook in self.pre_inference:
            prompt = hook(trial, prompt)
        return trial.model_copy(
            update={
                "prompt": trial.prompt.__class__(
                    id=trial.prompt.id,
                    family=trial.prompt.family,
                    variables=dict(trial.prompt.variables),
                    messages=prompt.messages,
                    follow_up_turns=prompt.follow_up_turns,
                ),
                "tools": prompt.tools,
                "mcp_servers": prompt.mcp_servers,
            }
        )

    def apply_post_inference(
        self,
        trial: TrialSpec,
        result: InferenceResult,
    ) -> InferenceResult:
        updated = result
        for hook in self.post_inference:
            updated = hook(trial, updated)
        return updated

    def apply_pre_extraction(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
    ) -> CandidateRecord:
        updated = candidate
        for hook in self.pre_extraction:
            updated = hook(trial, updated)
        return updated

    def apply_post_extraction(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
    ) -> CandidateRecord:
        updated = candidate
        for hook in self.post_extraction:
            updated = hook(trial, updated)
        return updated

    def apply_pre_eval(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
    ) -> CandidateRecord:
        updated = candidate
        for hook in self.pre_eval:
            updated = hook(trial, updated)
        return updated

    def apply_post_eval(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
    ) -> CandidateRecord:
        updated = candidate
        for hook in self.post_eval:
            updated = hook(trial, updated)
        return updated


@dataclass(frozen=True, slots=True)
class ResolvedExtractorStep:
    """One concrete extractor plus its configured invocation payload."""

    extractor_id: str
    config: Mapping[str, JSONValueType]
    extractor: Extractor


@dataclass(frozen=True, slots=True)
class ResolvedMetricStep:
    """One concrete metric instance resolved for execution."""

    metric_id: str
    config: Mapping[str, JSONValueType]
    metric: Metric | TrialMetric


@dataclass(frozen=True, slots=True)
class ResolvedGenerationPlugins:
    """Concrete dependencies needed for generation-stage execution."""

    engine: InferenceEngine
    hooks: ResolvedPipelineHooks


@dataclass(frozen=True, slots=True)
class ResolvedTransformPlugins:
    """Concrete dependencies needed for one output-transform stage."""

    transform: ResolvedOutputTransform
    extractors: tuple[ResolvedExtractorStep, ...]
    hooks: ResolvedPipelineHooks


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationPlugins:
    """Concrete dependencies needed for one evaluation stage."""

    evaluation: ResolvedEvaluation
    metrics: tuple[ResolvedMetricStep, ...]
    hooks: ResolvedPipelineHooks


@dataclass(slots=True)
class ResolvedTrialPlugins:
    """All concrete stage dependencies resolved once for one trial."""

    resolved_stages: ResolvedTaskStages
    generation: ResolvedGenerationPlugins | None
    hooks: ResolvedPipelineHooks = field(repr=False)
    output_transforms: tuple[ResolvedTransformPlugins, ...]
    evaluations: tuple[ResolvedEvaluationPlugins, ...]
    _judge_engine_resolver: JudgeEngineResolver = field(repr=False)
    _judge_engines: dict[str, InferenceEngine] = field(
        init=False, repr=False, default_factory=dict
    )
    _output_transforms_by_hash: dict[str, ResolvedTransformPlugins] = field(
        init=False, repr=False, default_factory=dict
    )
    _evaluations_by_hash: dict[str, ResolvedEvaluationPlugins] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._output_transforms_by_hash = {
            resolved.transform.transform_hash: resolved
            for resolved in self.output_transforms
        }
        self._evaluations_by_hash = {
            resolved.evaluation.evaluation_hash: resolved
            for resolved in self.evaluations
        }

    def output_transform_for(
        self, transform_hash: str
    ) -> ResolvedTransformPlugins | None:
        return self._output_transforms_by_hash.get(transform_hash)

    def evaluation_for(self, evaluation_hash: str) -> ResolvedEvaluationPlugins | None:
        return self._evaluations_by_hash.get(evaluation_hash)

    def create_judge_service(self) -> JudgeService:
        from themis.evaluation.judge_service import DefaultJudgeService

        return DefaultJudgeService(engine_resolver=self._resolve_cached_judge_engine)

    def _resolve_cached_judge_engine(self, provider: str) -> InferenceEngine:
        engine = self._judge_engines.get(provider)
        if engine is None:
            engine = self._judge_engine_resolver(provider)
            self._judge_engines[provider] = engine
        return engine


def resolve_trial_plugins(
    trial: TrialSpec,
    registry: PluginRegistry,
    *,
    resolved_stages: ResolvedTaskStages | None = None,
    required_stages: Sequence[ResolvedStage] | None = None,
) -> ResolvedTrialPlugins:
    """Resolve all concrete stage dependencies for one trial."""
    stages = resolved_stages or resolve_task_stages(trial.task)
    selected_stages = (
        _declared_trial_stages(trial)
        if required_stages is None
        else set(required_stages)
    )
    hooks = ResolvedPipelineHooks.from_registry(registry)
    generation = (
        resolve_generation_plugins(trial, registry, hooks=hooks)
        if trial.task.generation is not None and "generation" in selected_stages
        else None
    )
    return ResolvedTrialPlugins(
        resolved_stages=stages,
        generation=generation,
        hooks=hooks,
        output_transforms=tuple(
            resolve_transform_plugins(transform, registry, hooks=hooks)
            for transform in stages.output_transforms
        )
        if "transform" in selected_stages
        else (),
        evaluations=tuple(
            resolve_evaluation_plugins(evaluation, registry, hooks=hooks)
            for evaluation in stages.evaluations
        )
        if "evaluation" in selected_stages
        else (),
        _judge_engine_resolver=registry.get_inference_engine,
    )


def resolve_generation_plugins(
    trial: TrialSpec,
    registry: PluginRegistry,
    *,
    hooks: ResolvedPipelineHooks | None = None,
) -> ResolvedGenerationPlugins:
    """Resolve generation-stage dependencies for one trial."""
    return ResolvedGenerationPlugins(
        engine=registry.get_inference_engine(trial.model.provider),
        hooks=hooks or ResolvedPipelineHooks.from_registry(registry),
    )


def resolve_transform_plugins(
    transform: ResolvedOutputTransform,
    registry: PluginRegistry,
    *,
    hooks: ResolvedPipelineHooks | None = None,
) -> ResolvedTransformPlugins:
    """Resolve extractor dependencies for one output-transform stage."""
    return ResolvedTransformPlugins(
        transform=transform,
        extractors=tuple(
            ResolvedExtractorStep(
                extractor_id=extractor_ref.id,
                config=extractor_ref.config,
                extractor=registry.get_extractor(extractor_ref.id),
            )
            for extractor_ref in transform.spec.extractor_chain.extractors
        ),
        hooks=hooks or ResolvedPipelineHooks.from_registry(registry),
    )


def resolve_evaluation_plugins(
    evaluation: ResolvedEvaluation,
    registry: PluginRegistry,
    *,
    hooks: ResolvedPipelineHooks | None = None,
) -> ResolvedEvaluationPlugins:
    """Resolve metric dependencies for one evaluation stage."""
    metric_steps: list[ResolvedMetricStep] = []
    for metric_ref in evaluation.spec.metrics:
        coerced_metric_ref = _coerce_metric_ref(metric_ref)
        metric_steps.append(
            ResolvedMetricStep(
                metric_id=coerced_metric_ref.id,
                config=coerced_metric_ref.config,
                metric=_resolve_evaluation_metric(registry, coerced_metric_ref.id),
            )
        )
    return ResolvedEvaluationPlugins(
        evaluation=evaluation,
        metrics=tuple(metric_steps),
        hooks=hooks or ResolvedPipelineHooks.from_registry(registry),
    )


def _resolve_evaluation_metric(
    registry: PluginRegistry,
    metric_id: str,
) -> Metric | TrialMetric:
    metric = registry.get_metric(metric_id)
    if isinstance(metric, Metric) or isinstance(metric, TrialMetric):
        return metric
    if isinstance(metric, TraceMetric):
        raise TypeError(
            f"Metric '{metric_id}' is trace-only and cannot run in candidate evaluation."
        )
    raise TypeError(f"Metric '{metric_id}' is incompatible with evaluation execution.")


def _coerce_metric_ref(metric_ref: object) -> MetricRefSpec:
    if isinstance(metric_ref, MetricRefSpec):
        return metric_ref
    return MetricRefSpec.model_validate(metric_ref)


def _declared_trial_stages(trial: TrialSpec) -> set[ResolvedStage]:
    stages: set[ResolvedStage] = set()
    if trial.task.generation is not None:
        stages.add(RunStage.GENERATION)
    if trial.task.output_transforms:
        stages.add(RunStage.TRANSFORM)
    if trial.task.evaluations:
        stages.add(RunStage.EVALUATION)
    return stages
