from __future__ import annotations

import pytest

from themis.errors.exceptions import SpecValidationError
from themis.registry.compatibility import CompatibilityChecker
from themis.registry.plugin_registry import EngineCapabilities, PluginRegistry
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    ModelSpec,
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
            dataset=DatasetSpec(source="memory"),
            default_extractor_chain=ExtractorChainSpec(extractors=["json"]),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )


def test_compatibility_checker_reports_all_missing_trial_plugins() -> None:
    trial = _make_trial()
    checker = CompatibilityChecker()

    issues = checker.check_trial(trial, PluginRegistry())

    assert [issue.path for issue in issues] == [
        "model.provider",
        "task.default_extractor_chain.extractors[0]",
        "task.default_metrics[0]",
    ]


def test_compatibility_checker_passes_for_registered_trial_plugins() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(
            supports_response_format={"text", "json"},
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

    checker = CompatibilityChecker()

    assert checker.check_trial(_make_trial(), registry) == []


def test_compatibility_checker_raises_spec_validation_error_for_first_issue() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(),
    )
    checker = CompatibilityChecker()

    with pytest.raises(SpecValidationError, match="Extractor 'json' is not registered"):
        checker.validate_trial(_make_trial(), registry)


def test_compatibility_checker_rejects_plugin_api_major_mismatch() -> None:
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

    checker = CompatibilityChecker()
    issues = checker.check_trial(_make_trial(), registry)

    assert any("plugin_api" in issue.message for issue in issues)


def test_compatibility_checker_rejects_unsupported_response_format_and_logprobs() -> (
    None
):
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(
            supports_response_format={"text"},
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

    checker = CompatibilityChecker()
    issues = checker.check_trial(
        _make_trial().model_copy(
            update={"params": InferenceParamsSpec(response_format="json", logprobs=5)}
        ),
        registry,
    )

    assert [issue.path for issue in issues] == [
        "params.response_format",
        "params.logprobs",
    ]
