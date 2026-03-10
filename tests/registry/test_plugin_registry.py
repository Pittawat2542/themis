import pytest
from themis.contracts.protocols import InferenceResult
from themis.errors.exceptions import SpecValidationError
from themis.records.evaluation import MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.registry.plugin_registry import EngineCapabilities, PluginRegistry


class DummyEngine:
    def infer(self, trial, context, runtime):
        return InferenceResult(
            inference=InferenceRecord(spec_hash="inf_1", raw_text="42")
        )


class DummyMetric:
    def score(self, trial, candidate, context):
        return MetricScore(metric_id="exact_match", value=1.0)


class DummyExtractor:
    def extract(self, trial, candidate, config=None):
        return ExtractionRecord(
            spec_hash="ext_1", extractor_id="dummy", success=True, parsed_answer="42"
        )


class DummySelector:
    def select(self, candidates):
        return candidates[0] if candidates else None


class FirstHook:
    def post_inference(self, trial, result):
        return result


class SecondHook:
    def post_inference(self, trial, result):
        return result


def test_plugin_registry_registration():
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        DummyEngine,
        version="1.2.3",
        plugin_api="1.0",
        capabilities=EngineCapabilities(
            supports_response_format={"text", "json"},
            supports_logprobs=True,
            max_context_tokens=128_000,
        ),
    )
    registry.register_metric(
        "exact_match", DummyMetric, version="0.1.0", plugin_api="1.0"
    )
    registry.register_extractor(
        "dummy", DummyExtractor, version="0.1.0", plugin_api="1.0"
    )
    registry.register_candidate_selector(
        "top_1", DummySelector, version="0.1.0", plugin_api="1.0"
    )

    engine_registration = registry.get_inference_engine_registration("openai")
    assert engine_registration.version == "1.2.3"
    assert engine_registration.plugin_api == "1.0"
    assert engine_registration.capabilities.supports_logprobs is True
    assert isinstance(registry.get_inference_engine("openai"), DummyEngine)
    assert isinstance(registry.get_metric("exact_match"), DummyMetric)
    assert isinstance(registry.get_extractor("dummy"), DummyExtractor)
    assert isinstance(registry.get_candidate_selector("top_1"), DummySelector)

    with pytest.raises(SpecValidationError):
        registry.get_extractor("nonexistent")


def test_plugin_registry_orders_hooks_by_priority_and_registration_order():
    registry = PluginRegistry()
    first = FirstHook()
    second = SecondHook()

    registry.register_hook("second", second, priority=50)
    registry.register_hook("first", first, priority=50)

    assert [
        registration.name for registration in registry.iter_hook_registrations()
    ] == [
        "second",
        "first",
    ]
