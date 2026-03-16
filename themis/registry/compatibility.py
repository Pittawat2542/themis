"""Compatibility checks for providers, extractors, metrics, and run stages."""

from __future__ import annotations

from collections.abc import Collection
import warnings
from typing import TYPE_CHECKING, TypeAlias

from themis.errors import SpecValidationError
from themis.registry.plugin_registry import PluginRegistry, SUPPORTED_PLUGIN_API_MAJOR
from themis.specs.experiment import TrialSpec
from themis.specs.foundational import EvaluationSpec, OutputTransformSpec
from themis.types.enums import ErrorCode, RunStage
from themis.types.issues import Issue

if TYPE_CHECKING:
    from themis.orchestration.resolved_plugins import ResolvedTrialPlugins

TrialStage: TypeAlias = RunStage


class CompatibilityChecker:
    """Deprecated adapter over the functional compatibility helpers."""

    def __init__(self, registry: PluginRegistry) -> None:
        warnings.warn(
            "CompatibilityChecker is deprecated; use the functional helpers in "
            "themis.registry.compatibility instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.registry = registry

    def check_generation_trial(self, trial: TrialSpec) -> list[Issue]:
        """Returns generation-stage compatibility issues for ``trial``."""

        return check_generation_trial(trial, self.registry)

    def validate_generation_trial(self, trial: TrialSpec) -> None:
        """Raises when generation-stage compatibility checks fail."""

        validate_generation_trial(trial, self.registry)

    def check_output_transform(
        self,
        output_transform: OutputTransformSpec,
    ) -> list[Issue]:
        """Returns compatibility issues for one output transform."""

        return check_output_transform(output_transform, self.registry)

    def validate_output_transform(
        self,
        output_transform: OutputTransformSpec,
    ) -> None:
        """Raises when an output transform is incompatible."""

        validate_output_transform(output_transform, self.registry)

    def check_evaluation_spec(self, evaluation: EvaluationSpec) -> list[Issue]:
        """Returns compatibility issues for one evaluation spec."""

        return check_evaluation_spec(evaluation, self.registry)

    def validate_evaluation_spec(self, evaluation: EvaluationSpec) -> None:
        """Raises when one evaluation spec is incompatible."""

        validate_evaluation_spec(evaluation, self.registry)

    def check_trial(self, trial: TrialSpec) -> list[Issue]:
        """Returns compatibility issues across all declared trial stages."""

        return check_trial(trial, self.registry)

    def validate_trial(self, trial: TrialSpec) -> None:
        """Raises when any declared stage of ``trial`` is incompatible."""

        validate_trial(trial, self.registry)

    def check_trial_for_stages(
        self,
        trial: TrialSpec,
        *,
        stages: Collection[TrialStage],
    ) -> list[Issue]:
        """Returns compatibility issues only for the requested stages."""

        return check_trial_for_stages(trial, self.registry, stages=stages)

    def validate_trial_for_stages(
        self,
        trial: TrialSpec,
        *,
        stages: Collection[TrialStage],
    ) -> None:
        """Raises when the requested stages of ``trial`` are incompatible."""

        validate_trial_for_stages(trial, self.registry, stages=stages)

    def resolve_trial_plugins(self, trial: TrialSpec) -> ResolvedTrialPlugins:
        """Validates ``trial`` and resolves its concrete plugin registrations."""

        return resolve_trial_plugins(trial, self.registry)


def check_generation_trial(trial: TrialSpec, registry: PluginRegistry) -> list[Issue]:
    """Return plugin compatibility issues for the generation stage."""
    issues: list[Issue] = []

    if not registry.has_inference_engine(trial.model.provider):
        issues.append(
            Issue(
                code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                path="model.provider",
                message=f"Provider '{trial.model.provider}' is not registered.",
                suggestion="Register an inference engine for the requested provider before execution.",
            )
        )
        return issues

    engine_registration = registry.get_inference_engine_registration(
        trial.model.provider
    )
    plugin_api_major = _plugin_api_major(engine_registration.plugin_api)
    if plugin_api_major != SUPPORTED_PLUGIN_API_MAJOR:
        issues.append(
            Issue(
                code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                path="model.provider",
                message=(
                    f"Provider '{trial.model.provider}' plugin_api={engine_registration.plugin_api} "
                    f"is incompatible with supported major {SUPPORTED_PLUGIN_API_MAJOR}.x."
                ),
                suggestion="Register a provider plugin built against the supported plugin API major version.",
            )
        )

    response_format = trial.params.response_format
    if (
        response_format is not None
        and response_format
        not in engine_registration.capabilities.supports_response_format
    ):
        issues.append(
            Issue(
                code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                path="params.response_format",
                message=(
                    f"Provider '{trial.model.provider}' does not support "
                    f"response_format='{response_format.value}'."
                ),
                suggestion="Select a supported response format or switch providers.",
            )
        )

    if trial.params.logprobs and not engine_registration.capabilities.supports_logprobs:
        issues.append(
            Issue(
                code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                path="params.logprobs",
                message=f"Provider '{trial.model.provider}' does not support logprobs.",
                suggestion="Disable logprobs or switch to an engine that supports them.",
            )
        )

    estimator = engine_registration.prompt_token_estimator
    max_context_tokens = engine_registration.capabilities.max_context_tokens
    if estimator is not None and max_context_tokens is not None:
        estimated_prompt_tokens = estimator(trial)
        if estimated_prompt_tokens > max_context_tokens:
            issues.append(
                Issue(
                    code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                    path="prompt.messages",
                    message=(
                        f"Provider '{trial.model.provider}' estimated prompt tokens "
                        f"({estimated_prompt_tokens}) exceed max_context_tokens={max_context_tokens}."
                    ),
                    suggestion="Shorten the prompt or switch to a provider with a larger context window.",
                )
            )

    return issues


def validate_generation_trial(trial: TrialSpec, registry: PluginRegistry) -> None:
    """Raise `SpecValidationError` when generation compatibility fails."""
    _raise_on_first_issue(check_generation_trial(trial, registry))


def check_output_transform(
    output_transform: OutputTransformSpec,
    registry: PluginRegistry,
) -> list[Issue]:
    """Return plugin compatibility issues for one output-transform stage."""
    issues: list[Issue] = []

    for index, extractor_ref in enumerate(output_transform.extractor_chain.extractors):
        extractor_id = extractor_ref.id
        if not registry.has_extractor(extractor_id):
            issues.append(
                Issue(
                    code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                    path=f"output_transform.extractor_chain.extractors[{index}]",
                    message=f"Extractor '{extractor_id}' is not registered.",
                    suggestion="Register the extractor or remove it from the output transform.",
                )
            )
            continue

        extractor_registration = registry.get_extractor_registration(extractor_id)
        if (
            _plugin_api_major(extractor_registration.plugin_api)
            != SUPPORTED_PLUGIN_API_MAJOR
        ):
            issues.append(
                Issue(
                    code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                    path=f"output_transform.extractor_chain.extractors[{index}]",
                    message=(
                        f"Extractor '{extractor_id}' plugin_api={extractor_registration.plugin_api} "
                        f"is incompatible with supported major {SUPPORTED_PLUGIN_API_MAJOR}.x."
                    ),
                    suggestion="Register an extractor built against the supported plugin API major version.",
                )
            )

    return issues


def validate_output_transform(
    output_transform: OutputTransformSpec,
    registry: PluginRegistry,
) -> None:
    """Raise `SpecValidationError` when one output transform is incompatible."""
    _raise_on_first_issue(check_output_transform(output_transform, registry))


def check_evaluation_spec(
    evaluation: EvaluationSpec,
    registry: PluginRegistry,
) -> list[Issue]:
    """Return plugin compatibility issues for one evaluation stage."""
    issues: list[Issue] = []

    for index, metric_id in enumerate(evaluation.metrics):
        if not registry.has_metric(metric_id):
            issues.append(
                Issue(
                    code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                    path=f"evaluation.metrics[{index}]",
                    message=f"Metric '{metric_id}' is not registered.",
                    suggestion="Register the metric or remove it from the evaluation.",
                )
            )
            continue

        metric_registration = registry.get_metric_registration(metric_id)
        if (
            _plugin_api_major(metric_registration.plugin_api)
            != SUPPORTED_PLUGIN_API_MAJOR
        ):
            issues.append(
                Issue(
                    code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                    path=f"evaluation.metrics[{index}]",
                    message=(
                        f"Metric '{metric_id}' plugin_api={metric_registration.plugin_api} "
                        f"is incompatible with supported major {SUPPORTED_PLUGIN_API_MAJOR}.x."
                    ),
                    suggestion="Register a metric built against the supported plugin API major version.",
                )
            )

    return issues


def validate_evaluation_spec(
    evaluation: EvaluationSpec,
    registry: PluginRegistry,
) -> None:
    """Raise `SpecValidationError` when one evaluation stage is incompatible."""
    _raise_on_first_issue(check_evaluation_spec(evaluation, registry))


def check_trial(trial: TrialSpec, registry: PluginRegistry) -> list[Issue]:
    """Return the full compatibility issue list for a trial spec."""
    return check_trial_for_stages(
        trial,
        registry,
        stages=_declared_trial_stages(trial),
    )


def validate_trial(trial: TrialSpec, registry: PluginRegistry) -> None:
    """Raise `SpecValidationError` when any stage of a trial is incompatible."""
    validate_trial_for_stages(
        trial,
        registry,
        stages=_declared_trial_stages(trial),
    )


def check_trial_for_stages(
    trial: TrialSpec,
    registry: PluginRegistry,
    *,
    stages: Collection[TrialStage],
) -> list[Issue]:
    """Return compatibility issues for only the requested trial stages."""

    stage_set = set(stages)
    issues: list[Issue] = []

    if "generation" in stage_set and trial.task.generation is not None:
        issues.extend(check_generation_trial(trial, registry))
    if "transform" in stage_set:
        for output_transform in trial.task.output_transforms:
            issues.extend(check_output_transform(output_transform, registry))
    if "evaluation" in stage_set:
        for evaluation in trial.task.evaluations:
            issues.extend(check_evaluation_spec(evaluation, registry))
    return issues


def validate_trial_for_stages(
    trial: TrialSpec,
    registry: PluginRegistry,
    *,
    stages: Collection[TrialStage],
) -> None:
    """Raise `SpecValidationError` when requested trial stages are incompatible."""

    _raise_on_first_issue(check_trial_for_stages(trial, registry, stages=stages))


def resolve_trial_plugins(
    trial: TrialSpec,
    registry: PluginRegistry,
) -> ResolvedTrialPlugins:
    """Validate and resolve concrete stage plugins for one trial."""
    from themis.orchestration.resolved_plugins import (
        resolve_trial_plugins as resolve_plugins,
    )

    validate_trial(trial, registry)
    return resolve_plugins(trial, registry)


def _raise_on_first_issue(issues: list[Issue]) -> None:
    if not issues:
        return

    first_issue = issues[0]
    raise SpecValidationError(
        code=ErrorCode.PLUGIN_INCOMPATIBLE,
        message=first_issue.message,
        details={
            "path": first_issue.path,
            "suggestion": first_issue.suggestion,
        },
    )


def _plugin_api_major(plugin_api: str) -> int | None:
    try:
        return int(plugin_api.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError):
        return None


def _declared_trial_stages(trial: TrialSpec) -> tuple[TrialStage, ...]:
    stages: list[TrialStage] = []
    if trial.task.generation is not None:
        stages.append(RunStage.GENERATION)
    if trial.task.output_transforms:
        stages.append(RunStage.TRANSFORM)
    if trial.task.evaluations:
        stages.append(RunStage.EVALUATION)
    return tuple(stages)
