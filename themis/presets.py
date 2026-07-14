"""Public named presets over the canonical experiment and runtime options."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Literal

from pydantic import Field

from themis.api import Experiment, RunOptions
from themis.core.base import FrozenModel, JSONValue
from themis.core.config import (
    GeneratorComponent,
    JudgeModelComponent,
    MetricComponent,
    ParserComponent,
    ReducerComponent,
    SelectorComponent,
)


class RuntimePreset(FrozenModel):
    preset_id: str
    options: RunOptions
    description: str = ""
    kind: Literal["runtime"] = "runtime"


class GenerationPreset(FrozenModel):
    preset_id: str
    generator: GeneratorComponent | None = None
    samples: int | None = None
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None
    description: str = ""
    kind: Literal["generation"] = "generation"


class EvaluationPreset(FrozenModel):
    preset_id: str
    metrics: list[MetricComponent] = Field(default_factory=list)
    parser: ParserComponent | None = None
    parser_views: dict[str, ParserComponent] = Field(default_factory=dict)
    judge_models: list[JudgeModelComponent] = Field(default_factory=list)
    judge_options: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_options: dict[str, JSONValue] = Field(default_factory=dict)
    description: str = ""
    kind: Literal["evaluation"] = "evaluation"


class ExperimentPreset(FrozenModel):
    preset_id: str
    generation: GenerationPreset | None = None
    evaluation: EvaluationPreset | None = None
    runtime: RuntimePreset | None = None
    description: str = ""
    kind: Literal["experiment"] = "experiment"


type Preset = RuntimePreset | GenerationPreset | EvaluationPreset | ExperimentPreset


class PresetApplication(FrozenModel):
    """Resolved identity and runtime values produced by applying a preset."""

    experiment: Experiment
    options: RunOptions
    preset_ids: list[str] = Field(default_factory=list)


def list_presets(*, kind: str | None = None) -> list[str]:
    """List known preset identifiers, optionally filtered by kind."""

    return sorted(
        preset_id
        for preset_id, preset in _preset_registry().items()
        if kind is None or preset.kind == kind
    )


def get_preset(preset_id: str) -> Preset:
    """Return one preset by id with close-match suggestions on failure."""

    presets = _preset_registry()
    try:
        return presets[preset_id]
    except KeyError as exc:
        suggestions = get_close_matches(preset_id, presets, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown preset: {preset_id}.{hint}".rstrip(".")) from exc


def apply_preset(
    experiment: Experiment,
    preset_id: str,
    *,
    options: RunOptions | None = None,
) -> PresetApplication:
    """Resolve one preset over an experiment and runtime options."""

    return _apply(
        PresetApplication(experiment=experiment, options=options or RunOptions()),
        get_preset(preset_id),
    )


def _apply(application: PresetApplication, preset: Preset) -> PresetApplication:
    experiment = application.experiment
    options = application.options
    preset_ids = [*application.preset_ids]

    if isinstance(preset, RuntimePreset):
        options = preset.options
    elif isinstance(preset, GenerationPreset):
        generation_update: dict[str, object] = {
            key: value
            for key, value in {
                "generator": preset.generator,
                "samples": preset.samples,
                "selector": preset.selector,
                "reducer": preset.reducer,
            }.items()
            if value is not None
        }
        experiment = experiment.model_copy(
            update={
                "generation": experiment.generation.model_copy(update=generation_update)
            }
        )
    elif isinstance(preset, EvaluationPreset):
        evaluation_update: dict[str, object] = {}
        for key in ("metrics", "parser_views", "judge_models"):
            value = getattr(preset, key)
            if value:
                evaluation_update[key] = value
        if preset.parser is not None:
            evaluation_update["parser"] = preset.parser
            evaluation_update["parser_views"] = {}
        if preset.judge_options:
            evaluation_update["judge_options"] = {
                **experiment.evaluation.judge_options,
                **preset.judge_options,
            }
        if preset.workflow_options:
            evaluation_update["workflow_options"] = {
                **experiment.evaluation.workflow_options,
                **preset.workflow_options,
            }
        experiment = experiment.model_copy(
            update={
                "evaluation": experiment.evaluation.model_copy(update=evaluation_update)
            }
        )
    else:
        for child in (preset.generation, preset.evaluation, preset.runtime):
            if child is not None:
                child_application = _apply(
                    PresetApplication(
                        experiment=experiment,
                        options=options,
                        preset_ids=preset_ids,
                    ),
                    child,
                )
                experiment = child_application.experiment
                options = child_application.options
                preset_ids = list(child_application.preset_ids)

    preset_ids.append(preset.preset_id)
    experiment = experiment.model_copy(
        update={
            "metadata": {
                **experiment.metadata,
                f"themis.preset.{preset.preset_id}": "true",
            }
        }
    )
    return PresetApplication(
        experiment=experiment,
        options=options,
        preset_ids=preset_ids,
    )


def _preset_registry() -> dict[str, Preset]:
    runtime_fast = RuntimePreset(
        preset_id="runtime/local-fast",
        options=RunOptions(max_concurrency=32),
        description="High-concurrency local runtime defaults.",
    )
    runtime_careful = RuntimePreset(
        preset_id="runtime/local-careful",
        options=RunOptions(max_concurrency=2, store_retry_attempts=8),
        description="Lower-concurrency local runtime defaults.",
    )
    best_of_n = GenerationPreset(
        preset_id="candidate/best-of-n",
        samples=2,
        selector="builtin/best_of_n",
        description="Generate two candidates and select the best candidate.",
    )
    judge_rubric = EvaluationPreset(
        preset_id="judge/demo-rubric",
        metrics=["builtin/llm_rubric"],
        parser="builtin/text",
        judge_models=["builtin/demo_judge"],
        workflow_options={"rubric": "Judge whether the answer is correct."},
        description="Demo LLM-rubric evaluation using the builtin demo judge.",
    )
    code_local = ExperimentPreset(
        preset_id="code/local-subprocess",
        generation=GenerationPreset(
            preset_id="code/local-subprocess:generation", samples=1
        ),
        evaluation=EvaluationPreset(
            preset_id="code/local-subprocess:evaluation",
            metrics=["builtin/humaneval_pass_rate"],
            parser="builtin/code_text",
        ),
        runtime=runtime_careful,
        description="Local subprocess-oriented code benchmark defaults.",
    )
    baseline = ExperimentPreset(
        preset_id="baseline/demo",
        generation=GenerationPreset(
            preset_id="baseline/demo:generation",
            generator="builtin/demo_generator",
            samples=1,
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationPreset(
            preset_id="baseline/demo:evaluation",
            metrics=["builtin/exact_match"],
            parser="builtin/json_identity",
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


__all__ = [
    "EvaluationPreset",
    "ExperimentPreset",
    "GenerationPreset",
    "Preset",
    "PresetApplication",
    "RuntimePreset",
    "apply_preset",
    "get_preset",
    "list_presets",
]
