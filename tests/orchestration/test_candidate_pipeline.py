import pytest
from themis.orchestration.candidate_pipeline import execute_candidate_pipeline
from themis.registry.plugin_registry import EngineCapabilities, PluginRegistry
from themis.specs.experiment import (
    TrialSpec,
    PromptTemplateSpec,
    InferenceParamsSpec,
    RuntimeContext,
)
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    ModelSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records.inference import InferenceRecord
from themis.records.extraction import ExtractionRecord
from themis.records.evaluation import MetricScore
from themis.types.enums import RecordStatus


class MockInferenceEngine:
    def infer(self, trial, context, runtime):
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash="inf_hash",
                raw_text=trial.prompt.messages[0]["content"]
                if trial.prompt.messages
                else "The answer is 42.",
            )
        )


class MockExtractor:
    def extract(self, trial, candidate, config=None):
        text = candidate.inference.raw_text
        needle = "42"
        if isinstance(config, dict):
            needle = str(config.get("needle", needle))
        if needle in text:
            return ExtractionRecord(
                spec_hash="ext1", extractor_id="mock", success=True, parsed_answer="42"
            )
        return ExtractionRecord(spec_hash="ext1", extractor_id="mock", success=False)


class MockMetric:
    def score(self, trial, candidate, context):
        if candidate.extractions and candidate.extractions[-1].success:
            return MetricScore(metric_id="em", value=1.0)
        return MetricScore(metric_id="em", value=0.0)


@pytest.fixture
def registry():
    reg = PluginRegistry()
    reg.register_inference_engine(
        "openai",
        MockInferenceEngine,
        version="1.0.0",
        plugin_api="1.0",
        capabilities=EngineCapabilities(
            supports_response_format={"text", "json"},
            supports_logprobs=True,
            max_context_tokens=128_000,
        ),
    )
    reg.register_extractor("mock", MockExtractor, version="1.0.0", plugin_api="1.0")
    reg.register_metric("em", MockMetric, version="1.0.0", plugin_api="1.0")
    return reg


@pytest.fixture
def trial():
    return TrialSpec(
        trial_id="test_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="t1",
            dataset=DatasetSpec(source="memory"),
            default_extractor_chain=ExtractorChainSpec(
                extractors=[{"id": "mock", "config": {"needle": "42"}}]
            ),
            default_metrics=["em"],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )


def test_execute_candidate_pipeline_success(registry, trial):
    context = {}
    runtime = RuntimeContext(environment={"suite": "tests"})

    candidate = execute_candidate_pipeline(
        trial, registry, context, runtime, cand_index=0
    )

    if candidate.error:
        print(f"PIPELINE ERROR: {candidate.error.cause_chain}")

    assert candidate.status == RecordStatus.OK
    assert candidate.inference.raw_text == "The answer is 42."
    assert len(candidate.extractions) == 1
    assert candidate.extractions[0].success is True
    assert getattr(candidate.evaluation, "aggregate_scores")["em"] == 1.0


def test_execute_candidate_pipeline_no_extractor(registry, trial):
    trial_no_ext = TrialSpec(
        trial_id="test_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="t1",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["em"],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )

    candidate = execute_candidate_pipeline(
        trial_no_ext,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )

    if candidate.error:
        print(f"PIPELINE ERROR: {candidate.error.cause_chain}")

    assert candidate.status == RecordStatus.OK
    assert len(candidate.extractions) == 0
    # The metric should still run
    assert "em" in candidate.evaluation.aggregate_scores


def test_execute_candidate_pipeline_applies_hooks_in_order(registry, trial):
    class PromptRewriteHook:
        def pre_inference(self, trial, prompt):
            return prompt.model_copy(
                update={"messages": [{"role": "user", "content": "The answer is 43."}]}
            )

    class ExtractionHook:
        def pre_extraction(self, trial, candidate):
            return candidate.model_copy(
                update={
                    "inference": candidate.inference.model_copy(
                        update={"raw_text": "The answer is 43."}
                    )
                }
            )

        def post_extraction(self, trial, candidate):
            extraction = candidate.extractions[-1].model_copy(
                update={"warnings": ["normalized"]}
            )
            return candidate.model_copy(update={"extractions": [extraction]})

        def post_eval(self, trial, candidate):
            score = candidate.evaluation.metric_scores[0].model_copy(
                update={"value": 0.5}
            )
            evaluation = candidate.evaluation.model_copy(
                update={
                    "metric_scores": [score],
                    "aggregate_scores": {"em": 0.5},
                }
            )
            return candidate.model_copy(update={"evaluation": evaluation})

    registry.register_hook("prompt", PromptRewriteHook(), priority=50)
    registry.register_hook("extraction", ExtractionHook(), priority=100)

    candidate = execute_candidate_pipeline(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )

    assert candidate.status == RecordStatus.OK
    assert candidate.inference.raw_text == "The answer is 43."
    assert candidate.extractions[0].warnings == ["normalized"]
    assert candidate.evaluation.aggregate_scores["em"] == 0.5
