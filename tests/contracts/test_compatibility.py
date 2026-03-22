from __future__ import annotations

import themis.registry as registry_module
import themis.registry.compatibility as compatibility_module
import pytest

from themis.errors import SpecValidationError
from themis.registry.compatibility import (
    check_evaluation_spec,
    check_generation_trial,
    check_output_transform,
    check_trial,
    check_trial_for_stages,
    resolve_trial_plugins,
    validate_output_transform,
)
from themis.types.enums import DatasetSource, PromptRole, ResponseFormat, RunStage
from themis.registry.plugin_registry import EngineCapabilities, PluginRegistry
from themis.specs.experiment import (
    InferenceParamsSpec,
    PromptMessage,
    PromptTemplateSpec,
    PromptTurnSpec,
    TrialSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    McpServerSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)


class DummyEngine:
    def infer(
        self, trial, context, runtime
    ):  # pragma: no cover - compatibility tests do not execute it
        raise NotImplementedError


class DummyExtractor:
    def extract(
        self, trial, candidate, config=None
    ):  # pragma: no cover - compatibility tests do not execute it
        raise NotImplementedError


class DummyMetric:
    def score(
        self, trial, candidate, context
    ):  # pragma: no cover - compatibility tests do not execute it
        raise NotImplementedError


def _make_trial() -> TrialSpec:
    return TrialSpec(
        trial_id="trial_compatibility",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="json")]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(
                    name="exact_match_eval",
                    transform="json",
                    metrics=["exact_match"],
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )


def _make_transform() -> OutputTransformSpec:
    return OutputTransformSpec(
        name="json",
        extractor_chain=ExtractorChainSpec(extractors=[ExtractorRefSpec(id="json")]),
    )


def _make_evaluation() -> EvaluationSpec:
    return EvaluationSpec(
        name="exact_match_eval",
        transform="json",
        metrics=["exact_match"],
    )


def test_compatibility_checker_is_available_via_deprecated_adapter() -> None:
    compatibility_checker = compatibility_module.CompatibilityChecker

    assert registry_module.CompatibilityChecker is compatibility_checker

    with pytest.deprecated_call():
        checker = compatibility_checker(PluginRegistry())

    issues = checker.check_generation_trial(_make_trial())

    assert [issue.path for issue in issues] == ["model.provider"]


def test_compatibility_reports_all_missing_generation_plugins() -> None:
    trial = _make_trial()

    issues = check_generation_trial(trial, PluginRegistry())

    assert [issue.path for issue in issues] == ["model.provider"]


def test_compatibility_reports_missing_transform_plugins() -> None:
    issues = check_output_transform(_make_transform(), PluginRegistry())

    assert [issue.path for issue in issues] == [
        "output_transform.extractor_chain.extractors[0]"
    ]


def test_compatibility_reports_missing_evaluation_plugins() -> None:
    issues = check_evaluation_spec(_make_evaluation(), PluginRegistry())

    assert [issue.path for issue in issues] == ["evaluation.metrics[0]"]


def test_compatibility_passes_for_registered_stage_plugins() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(
            supports_response_format={ResponseFormat.TEXT, ResponseFormat.JSON},
            supports_logprobs=True,
            max_context_tokens=32_000,
        ),
    )
    registry.register_extractor(
        "json", DummyExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric(
        "exact_match", DummyMetric, version="1.0.0", plugin_api="1.0"
    )

    assert check_generation_trial(_make_trial(), registry) == []
    assert check_output_transform(_make_transform(), registry) == []
    assert check_evaluation_spec(_make_evaluation(), registry) == []


def test_registered_stage_plugins_can_be_resolved() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(),
    )
    registry.register_extractor(
        "json", DummyExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric(
        "exact_match", DummyMetric, version="1.0.0", plugin_api="1.0"
    )

    resolved = resolve_trial_plugins(_make_trial(), registry)

    assert resolved.generation is not None
    assert [step.extractor_id for step in resolved.output_transforms[0].extractors] == [
        "json"
    ]
    assert [step.metric_id for step in resolved.evaluations[0].metrics] == [
        "exact_match"
    ]


def test_compatibility_raises_spec_validation_error_for_first_transform_issue() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(),
    )
    with pytest.raises(SpecValidationError, match="Extractor 'json' is not registered"):
        validate_output_transform(_make_transform(), registry)


def test_compatibility_rejects_plugin_api_major_mismatch() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="2.0",
        capabilities=EngineCapabilities(),
    )
    registry.register_extractor(
        "json", DummyExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric(
        "exact_match", DummyMetric, version="1.0.0", plugin_api="1.0"
    )

    issues = check_generation_trial(_make_trial(), registry)

    assert any("plugin_api" in issue.message for issue in issues)


def test_compatibility_rejects_unsupported_response_format_and_logprobs() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(
            supports_response_format={ResponseFormat.JSON},
            supports_logprobs=False,
            max_context_tokens=4_096,
        ),
    )
    registry.register_extractor(
        "json", DummyExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric(
        "exact_match", DummyMetric, version="1.0.0", plugin_api="1.0"
    )

    issues = check_generation_trial(
        _make_trial().model_copy(
            update={
                "params": InferenceParamsSpec(
                    response_format=ResponseFormat.TEXT, logprobs=5
                )
            }
        ),
        registry,
    )

    assert [issue.path for issue in issues] == [
        "params.response_format",
        "params.logprobs",
    ]


def test_compatibility_rejects_mcp_for_engines_without_support() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(),
    )

    issues = check_generation_trial(
        _make_trial().model_copy(
            update={
                "mcp_servers": [
                    McpServerSpec(
                        id="dice",
                        server_label="dice",
                        server_url="https://dmcp-server.deno.dev/sse",
                        require_approval="never",
                    )
                ]
            }
        ),
        registry,
    )

    assert [issue.path for issue in issues] == ["mcp_servers"]


def test_compatibility_rejects_mcp_approval_required_runs() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(supports_mcp=True),
    )

    issues = check_generation_trial(
        _make_trial().model_copy(
            update={
                "mcp_servers": [
                    McpServerSpec(
                        id="calendar",
                        server_label="google_calendar",
                        connector_id="connector_googlecalendar",
                        require_approval="always",
                    )
                ]
            }
        ),
        registry,
    )

    assert [issue.path for issue in issues] == ["mcp_servers[0].require_approval"]


def test_compatibility_rejects_openai_mcp_with_follow_up_turns() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(supports_mcp=True),
    )

    issues = check_generation_trial(
        _make_trial().model_copy(
            update={
                "prompt": PromptTemplateSpec(
                    id="agent",
                    messages=[],
                    follow_up_turns=[
                        PromptTurnSpec(
                            messages=[
                                PromptMessage(
                                    role=PromptRole.USER,
                                    content="Continue.",
                                )
                            ]
                        )
                    ],
                ),
                "mcp_servers": [
                    McpServerSpec(
                        id="dice",
                        server_label="dice",
                        server_url="https://dmcp-server.deno.dev/sse",
                        require_approval="never",
                    )
                ],
            }
        ),
        registry,
    )

    assert [issue.path for issue in issues] == ["prompt.follow_up_turns"]


def test_compatibility_can_scope_validation_to_generation_only() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(),
    )

    assert (
        check_trial_for_stages(
            _make_trial(),
            registry,
            stages={RunStage.GENERATION},
        )
        == []
    )


def test_compatibility_skips_generation_for_transform_only_tasks() -> None:
    trial = TrialSpec(
        trial_id="trial_transform_only",
        model=ModelSpec(model_id="imported-model", provider="unregistered"),
        task=TaskSpec(
            task_id="transform-only",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="json")]
                    ),
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    registry = PluginRegistry()
    registry.register_extractor(
        "json", DummyExtractor, version="1.0.0", plugin_api="1.0"
    )

    assert check_trial(trial, registry) == []
