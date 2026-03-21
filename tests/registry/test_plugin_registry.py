import pytest
from themis.contracts.protocols import InferenceResult
from themis.errors import SpecValidationError
from themis.records.candidate import CandidateRecord
from themis.records.evaluation import MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.contracts.protocols import RenderedPrompt
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


class _NoOpHook:
    def pre_inference(self, trial, prompt: RenderedPrompt) -> RenderedPrompt:
        return prompt

    def post_inference(self, trial, result: InferenceResult) -> InferenceResult:
        return result

    def pre_extraction(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        return candidate

    def post_extraction(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        return candidate

    def pre_eval(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        return candidate

    def post_eval(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        return candidate


class FirstHook(_NoOpHook):
    def post_inference(self, trial, result):
        return result


class SecondHook(_NoOpHook):
    def post_inference(self, trial, result):
        return result


class PartialHook:
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
    registry.register_tool("search", {"callable": "opaque"})

    engine_registration = registry.get_inference_engine_registration("openai")
    assert engine_registration.version == "1.2.3"
    assert engine_registration.plugin_api == "1.0"
    assert engine_registration.capabilities.supports_logprobs is True
    assert isinstance(registry.get_inference_engine("openai"), DummyEngine)
    assert isinstance(registry.get_metric("exact_match"), DummyMetric)
    assert isinstance(registry.get_extractor("dummy"), DummyExtractor)
    assert registry.has_tool("search") is True
    assert registry.get_tool("search") == {"callable": "opaque"}

    with pytest.raises(SpecValidationError):
        registry.get_extractor("nonexistent")

    with pytest.raises(SpecValidationError):
        registry.get_tool("nonexistent")


def test_plugin_registry_removes_unused_extension_points():
    registry = PluginRegistry()

    assert not hasattr(registry, "register_candidate_selector")
    assert not hasattr(registry, "get_candidate_selector")
    assert not hasattr(registry, "register_repository")


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


def test_plugin_registry_rejects_partial_pipeline_hooks():
    registry = PluginRegistry()

    with pytest.raises(SpecValidationError, match="PipelineHook"):
        registry.register_hook("partial", PartialHook())


def test_plugin_registry_from_dict_registers_engines_metrics_extractors() -> None:
    registry = PluginRegistry.from_dict(
        {
            "engines": {"demo": DummyEngine},
            "metrics": {"exact_match": DummyMetric},
            "extractors": {"dummy": DummyExtractor},
        }
    )
    assert registry.has_inference_engine("demo")
    assert registry.has_metric("exact_match")
    assert registry.has_extractor("dummy")


def test_plugin_registry_from_dict_empty_mapping_gives_empty_registry() -> None:
    registry = PluginRegistry.from_dict({})
    assert not registry.has_inference_engine("anything")
    assert not registry.has_metric("anything")


def test_plugin_registry_from_dict_registers_judges() -> None:
    class DummyJudge:
        def judge(self, *args, **kwargs):
            pass

        def consume_audit_trail(self):
            return []

    registry = PluginRegistry.from_dict({"judges": {"llm-judge": DummyJudge}})
    assert registry.has_judge("llm-judge")


def test_plugin_registry_from_dict_raises_on_unknown_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        PluginRegistry.from_dict({"unknown_section": {"x": object()}})


def test_plugin_registry_from_dict_builtin_extractors_still_present() -> None:
    """Built-in extractors should remain available after from_dict construction."""
    registry = PluginRegistry.from_dict({"metrics": {"em": DummyMetric}})
    assert registry.has_extractor("first_number")
    assert registry.has_extractor("choice_letter")


def test_engine_capabilities_supports_seed_defaults_false() -> None:
    caps = EngineCapabilities()
    assert caps.supports_seed is False


def test_engine_capabilities_supports_seed_can_be_set_true() -> None:
    caps = EngineCapabilities(supports_seed=True)
    assert caps.supports_seed is True


def test_engine_capabilities_supports_seed_preserved_in_registration() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine(
        "seed-aware-engine",
        DummyEngine,
        capabilities=EngineCapabilities(supports_seed=True),
    )
    registration = registry.get_inference_engine_registration("seed-aware-engine")
    assert registration.capabilities.supports_seed is True


def test_engine_capabilities_no_seed_support_preserved_in_registration() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("seed-unaware-engine", DummyEngine)
    registration = registry.get_inference_engine_registration("seed-unaware-engine")
    assert registration.capabilities.supports_seed is False
