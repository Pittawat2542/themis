"""Named configuration presets for common Themis workflows."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Literal

from pydantic import Field

from themis.core.base import FrozenModel, JSONValue
from themis.core.config import (
    EvaluationConfig,
    GeneratorComponent,
    JudgeModelComponent,
    MetricComponent,
    ParserComponent,
    ParserView,
    ReducerComponent,
    RuntimeConfig,
    SelectorComponent,
    SessionConfig,
)
from themis.core.experiment import Experiment


class RuntimePreset(FrozenModel):
    """Preset that changes runtime provenance without changing run identity."""

    preset_id: str
    runtime: RuntimeConfig
    description: str = ""
    kind: Literal["runtime"] = "runtime"


class SessionPreset(FrozenModel):
    """Preset that changes session-stage logical experiment behavior."""

    preset_id: str
    generator: GeneratorComponent | None = None
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None
    description: str = ""
    kind: Literal["session"] = "session"


class EvaluationPreset(FrozenModel):
    """Preset that changes evaluation-stage logical experiment behavior."""

    preset_id: str
    metrics: list[MetricComponent] = Field(default_factory=list)
    parsers: list[ParserView | ParserComponent] = Field(default_factory=list)
    judge_models: list[JudgeModelComponent] = Field(default_factory=list)
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)
    description: str = ""
    kind: Literal["evaluation"] = "evaluation"


class ExperimentPreset(FrozenModel):
    """Preset that may apply session, evaluation, and runtime overlays."""

    preset_id: str
    session: SessionPreset | None = None
    evaluation: EvaluationPreset | None = None
    runtime: RuntimePreset | None = None
    description: str = ""
    kind: Literal["experiment"] = "experiment"


Preset = RuntimePreset | SessionPreset | EvaluationPreset | ExperimentPreset


def list_presets(*, kind: str | None = None) -> list[str]:
    """List known preset identifiers, optionally filtered by kind."""

    presets = _preset_registry()
    return sorted(
        preset_id
        for preset_id, preset in presets.items()
        if kind is None or preset.kind == kind
    )


def get_preset(preset_id: str) -> Preset:
    """Return one preset by id with close-match suggestions on failure."""

    presets = _preset_registry()
    try:
        return presets[preset_id]
    except KeyError as exc:
        suggestions = get_close_matches(preset_id, presets, n=3, cutoff=0.5)
        if suggestions:
            raise ValueError(
                f"Unknown preset: {preset_id}. Did you mean: {', '.join(suggestions)}?"
            ) from exc
        raise ValueError(f"Unknown preset: {preset_id}") from exc


def apply_preset(experiment: Experiment, preset_id: str) -> Experiment:
    """Return a new experiment with a preset applied."""

    return _apply_preset(experiment, get_preset(preset_id))


def _apply_preset(experiment: Experiment, preset: Preset) -> Experiment:
    if isinstance(preset, RuntimePreset):
        return _copy_experiment(
            experiment,
            runtime=preset.runtime,
            environment_metadata=_metadata_with_preset(experiment, preset.preset_id),
        )
    if isinstance(preset, SessionPreset):
        return _copy_experiment(
            experiment,
            generation=_apply_session_preset(experiment.generation, preset),
            environment_metadata=_metadata_with_preset(experiment, preset.preset_id),
        )
    if isinstance(preset, EvaluationPreset):
        return _copy_experiment(
            experiment,
            evaluation=_apply_evaluation_preset(experiment.evaluation, preset),
            environment_metadata=_metadata_with_preset(experiment, preset.preset_id),
        )
    updated = experiment
    for child in (preset.session, preset.evaluation, preset.runtime):
        if child is not None:
            updated = _apply_preset(updated, child)
    return _copy_experiment(
        updated,
        environment_metadata=_metadata_with_preset(updated, preset.preset_id),
    )


def _apply_session_preset(
    generation: SessionConfig, preset: SessionPreset
) -> SessionConfig:
    update: dict[str, object] = {}
    if preset.generator is not None:
        update["generator"] = preset.generator
    if preset.candidate_policy:
        update["candidate_policy"] = {
            **generation.candidate_policy,
            **preset.candidate_policy,
        }
    if preset.selector is not None:
        update["selector"] = preset.selector
    if preset.reducer is not None:
        update["reducer"] = preset.reducer
    return generation.model_copy(update=update)


def _apply_evaluation_preset(
    evaluation: EvaluationConfig, preset: EvaluationPreset
) -> EvaluationConfig:
    update: dict[str, object] = {}
    if preset.metrics:
        update["metrics"] = list(preset.metrics)
    if preset.parsers:
        update["parsers"] = list(preset.parsers)
    if preset.judge_models:
        update["judge_models"] = list(preset.judge_models)
    if preset.judge_config:
        update["judge_config"] = {**evaluation.judge_config, **preset.judge_config}
    if preset.workflow_overrides:
        update["workflow_overrides"] = {
            **evaluation.workflow_overrides,
            **preset.workflow_overrides,
        }
    return evaluation.model_copy(update=update)


def _copy_experiment(experiment: Experiment, **updates: object) -> Experiment:
    copied = experiment.model_copy(update=updates)
    copied._compiled_snapshot = None
    return copied


def _metadata_with_preset(experiment: Experiment, preset_id: str) -> dict[str, str]:
    return {
        **experiment.environment_metadata,
        f"themis.preset.{preset_id}": "true",
    }


def _preset_registry() -> dict[str, Preset]:
    runtime_fast = RuntimePreset(
        preset_id="runtime/local-fast",
        runtime=RuntimeConfig(max_concurrent_tasks=32),
        description="High-concurrency local runtime defaults.",
    )
    runtime_careful = RuntimePreset(
        preset_id="runtime/local-careful",
        runtime=RuntimeConfig(max_concurrent_tasks=2, store_retry_attempts=8),
        description="Lower-concurrency local runtime defaults.",
    )
    best_of_n = SessionPreset(
        preset_id="candidate/best-of-n",
        candidate_policy={"num_samples": 2},
        selector="builtin/best_of_n",
        description="Generate two candidates and select the best candidate.",
    )
    judge_rubric = EvaluationPreset(
        preset_id="judge/demo-rubric",
        metrics=["builtin/llm_rubric"],
        parsers=["builtin/text"],
        judge_models=["builtin/demo_judge"],
        workflow_overrides={"rubric": "Judge whether the answer is correct."},
        description="Demo LLM-rubric evaluation using the builtin demo judge.",
    )
    code_local = ExperimentPreset(
        preset_id="code/local-subprocess",
        session=SessionPreset(
            preset_id="code/local-subprocess:session",
            candidate_policy={"num_samples": 1},
        ),
        evaluation=EvaluationPreset(
            preset_id="code/local-subprocess:evaluation",
            metrics=["builtin/humaneval_pass_rate"],
            parsers=["builtin/code_text"],
        ),
        runtime=runtime_careful,
        description="Local subprocess-oriented code benchmark defaults.",
    )
    baseline = ExperimentPreset(
        preset_id="baseline/demo",
        session=SessionPreset(
            preset_id="baseline/demo:session",
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationPreset(
            preset_id="baseline/demo:evaluation",
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        runtime=runtime_fast,
        description="Deterministic local demo baseline.",
    )
    return {
        preset.preset_id: preset
        for preset in (
            baseline,
            best_of_n,
            code_local,
            judge_rubric,
            runtime_careful,
            runtime_fast,
        )
    }
