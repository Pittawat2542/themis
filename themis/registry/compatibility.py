from __future__ import annotations

from themis.errors.exceptions import SpecValidationError
from themis.registry.plugin_registry import SUPPORTED_PLUGIN_API_MAJOR
from themis.specs.experiment import TrialSpec
from themis.types.enums import ErrorCode
from themis.types.issues import Issue


class CompatibilityChecker:
    """Pure compatibility validation over a trial spec and registry snapshot."""

    def check_trial(self, trial: TrialSpec, registry) -> list[Issue]:
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
        else:
            engine_registration = registry.get_inference_engine_registration(
                trial.model.provider
            )
            plugin_api_major = self._plugin_api_major(engine_registration.plugin_api)
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
                            f"Provider '{trial.model.provider}' does not support response_format='{response_format}'."
                        ),
                        suggestion="Select a supported response format or switch providers.",
                    )
                )

            if (
                trial.params.logprobs
                and not engine_registration.capabilities.supports_logprobs
            ):
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

        extractor_chain = trial.task.default_extractor_chain
        if extractor_chain:
            for index, extractor_ref in enumerate(extractor_chain.extractors):
                extractor_id = extractor_ref.id
                if not registry.has_extractor(extractor_id):
                    issues.append(
                        Issue(
                            code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                            path=f"task.default_extractor_chain.extractors[{index}]",
                            message=f"Extractor '{extractor_id}' is not registered.",
                            suggestion="Register the extractor or remove it from the task defaults.",
                        )
                    )
                    continue
                extractor_registration = registry.get_extractor_registration(
                    extractor_id
                )
                if (
                    self._plugin_api_major(extractor_registration.plugin_api)
                    != SUPPORTED_PLUGIN_API_MAJOR
                ):
                    issues.append(
                        Issue(
                            code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                            path=f"task.default_extractor_chain.extractors[{index}]",
                            message=(
                                f"Extractor '{extractor_id}' plugin_api={extractor_registration.plugin_api} "
                                f"is incompatible with supported major {SUPPORTED_PLUGIN_API_MAJOR}.x."
                            ),
                            suggestion="Register an extractor built against the supported plugin API major version.",
                        )
                    )

        for index, metric_id in enumerate(trial.task.default_metrics):
            if not registry.has_metric(metric_id):
                issues.append(
                    Issue(
                        code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                        path=f"task.default_metrics[{index}]",
                        message=f"Metric '{metric_id}' is not registered.",
                        suggestion="Register the metric or remove it from the task defaults.",
                    )
                )
                continue
            metric_registration = registry.get_metric_registration(metric_id)
            if (
                self._plugin_api_major(metric_registration.plugin_api)
                != SUPPORTED_PLUGIN_API_MAJOR
            ):
                issues.append(
                    Issue(
                        code=ErrorCode.PLUGIN_INCOMPATIBLE.value,
                        path=f"task.default_metrics[{index}]",
                        message=(
                            f"Metric '{metric_id}' plugin_api={metric_registration.plugin_api} "
                            f"is incompatible with supported major {SUPPORTED_PLUGIN_API_MAJOR}.x."
                        ),
                        suggestion="Register a metric built against the supported plugin API major version.",
                    )
                )

        return issues

    def validate_trial(self, trial: TrialSpec, registry) -> None:
        issues = self.check_trial(trial, registry)
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

    def _plugin_api_major(self, plugin_api: str) -> int | None:
        try:
            return int(plugin_api.split(".", maxsplit=1)[0])
        except (AttributeError, ValueError):
            return None
