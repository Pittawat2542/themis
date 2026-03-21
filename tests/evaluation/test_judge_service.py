import pytest

from themis.evaluation.judge_service import DefaultJudgeService
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import (
    InferenceParamsSpec,
    PromptTemplateSpec,
    RuntimeContext,
)
from themis.specs.foundational import JudgeInferenceSpec, ModelSpec
from themis.records.candidate import CandidateRecord
from themis.records.inference import InferenceRecord


class MockJudgeInferenceEngine:
    def __init__(self):
        self.params_seen: list[InferenceParamsSpec] = []
        self.trial_ids_seen: list[str] = []

    def infer(self, trial, dataset_context, runtime_context):
        assert isinstance(runtime_context, RuntimeContext)
        self.params_seen.append(trial.params)
        self.trial_ids_seen.append(trial.trial_id)
        return InferenceRecord(spec_hash="judge_inf_hash", raw_text="SCORE: 5/5")


def _resolver_from_registry(registry: PluginRegistry):
    return registry.get_inference_engine


def test_default_judge_service_rejects_registry_service_locator():
    registry = PluginRegistry()

    with pytest.raises(TypeError):
        DefaultJudgeService(registry)


def test_default_judge_service():
    registry = PluginRegistry()
    engine = MockJudgeInferenceEngine()
    registry.register_inference_engine("judge_provider", engine)

    judge_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(registry)
    )

    # Setup inputs
    parent_cand = CandidateRecord(spec_hash="parent_cand_123")
    judge_spec = JudgeInferenceSpec(
        model=ModelSpec(model_id="judge-model", provider="judge_provider")
    )
    prompt = PromptTemplateSpec(messages=[{"role": "user", "content": "Rate this."}])
    runtime = {"task_spec": None, "dataset_context": {}}

    # Execute judge
    inf_record = judge_service.judge(
        metric_id="test_judge_metric",
        parent_candidate=parent_cand,
        judge_spec=judge_spec,
        prompt=prompt,
        runtime=runtime,
    )

    assert inf_record.raw_text == "SCORE: 5/5"
    assert engine.params_seen == [InferenceParamsSpec()]

    # Audit trail fetching
    trail = judge_service.consume_audit_trail("parent_cand_123")
    assert trail is not None
    assert trail.candidate_hash == "parent_cand_123"
    assert len(trail.judge_calls) == 1

    call = trail.judge_calls[0]
    assert call.metric_id == "test_judge_metric"
    assert call.inference.raw_text == "SCORE: 5/5"
    assert call.judge_spec == judge_spec

    # Make sure it clears out calls for next candidate
    assert judge_service.consume_audit_trail("parent_cand_123") is None


def test_default_judge_service_multisample():
    registry = PluginRegistry()
    registry.register_inference_engine("judge_provider", MockJudgeInferenceEngine())
    judge_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(registry)
    )

    parent_cand = CandidateRecord(spec_hash="parent_cand_123")
    judge_spec = JudgeInferenceSpec(
        model=ModelSpec(model_id="judge-model", provider="judge_provider")
    )
    prompt = PromptTemplateSpec(messages=[{"role": "user", "content": "Rate this."}])

    import time

    # Call judge twice for the same candidate and metric (multi-sample judging)
    judge_service.judge(
        "test_metric",
        parent_cand,
        judge_spec,
        prompt,
        {"task_spec": None, "dataset_context": {}},
    )
    time.sleep(0.001)  # ensure time_ns advances across OS variants
    judge_service.judge(
        "test_metric",
        parent_cand,
        judge_spec,
        prompt,
        {"task_spec": None, "dataset_context": {}},
    )

    trail = judge_service.consume_audit_trail("parent_cand_123")
    assert len(trail.judge_calls) == 2


def test_default_judge_service_concurrency():
    registry = PluginRegistry()
    registry.register_inference_engine("judge_provider", MockJudgeInferenceEngine())
    judge_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(registry)
    )

    cand1 = CandidateRecord(spec_hash="cand_1")
    cand2 = CandidateRecord(spec_hash="cand_2")
    judge_spec = JudgeInferenceSpec(
        model=ModelSpec(model_id="j", provider="judge_provider")
    )
    prompt = PromptTemplateSpec(messages=[])

    # Simulate concurrent candidate pipelines hitting the judge service interleavingly
    judge_service.judge("m1", cand1, judge_spec, prompt, {"dataset_context": {}})
    judge_service.judge("m1", cand2, judge_spec, prompt, {"dataset_context": {}})
    judge_service.judge("m1", cand1, judge_spec, prompt, {"dataset_context": {}})

    # Audit trail for cand2 should ONLY have 1 call, and cand1 should have 2
    trail2 = judge_service.consume_audit_trail("cand_2")
    assert len(trail2.judge_calls) == 1

    trail1 = judge_service.consume_audit_trail("cand_1")
    assert len(trail1.judge_calls) == 2


def test_default_judge_service_uses_explicit_judge_params():
    registry = PluginRegistry()
    engine = MockJudgeInferenceEngine()
    registry.register_inference_engine("judge_provider", engine)
    judge_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(registry)
    )

    judge_service.judge(
        metric_id="temperature_metric",
        parent_candidate=CandidateRecord(spec_hash="parent_cand_123"),
        judge_spec=JudgeInferenceSpec(
            model=ModelSpec(model_id="judge-model", provider="judge_provider"),
            params={"temperature": 0.7, "max_tokens": 16},
        ),
        prompt=PromptTemplateSpec(messages=[{"role": "user", "content": "Rate this."}]),
        runtime={"dataset_context": {}},
    )

    assert engine.params_seen == [InferenceParamsSpec(temperature=0.7, max_tokens=16)]


def test_default_judge_service_derives_deterministic_trial_ids_and_seeds():
    registry = PluginRegistry()
    engine = MockJudgeInferenceEngine()
    registry.register_inference_engine("judge_provider", engine)
    judge_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(registry)
    )

    runtime_context = RuntimeContext(candidate_seed=101)
    for _ in range(2):
        judge_service.judge(
            metric_id="test_metric",
            parent_candidate=CandidateRecord(spec_hash="parent_cand_123"),
            judge_spec=JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="judge_provider")
            ),
            prompt=PromptTemplateSpec(
                messages=[{"role": "user", "content": "Rate this."}]
            ),
            runtime={
                "dataset_context": {},
                "runtime_context": runtime_context,
            },
        )

    assert engine.trial_ids_seen == [
        "judge_parent_cand_123_test_metric_0",
        "judge_parent_cand_123_test_metric_1",
    ]
    first_run_seeds = [params.seed for params in engine.params_seen]
    assert first_run_seeds[0] is not None
    assert first_run_seeds[1] is not None
    assert first_run_seeds[0] != first_run_seeds[1]
    trail = judge_service.consume_audit_trail("parent_cand_123")
    assert trail is not None
    assert [
        call.judge_spec.params.seed for call in trail.judge_calls
    ] == first_run_seeds

    second_engine = MockJudgeInferenceEngine()
    second_registry = PluginRegistry()
    second_registry.register_inference_engine("judge_provider", second_engine)
    second_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(second_registry)
    )
    for _ in range(2):
        second_service.judge(
            metric_id="test_metric",
            parent_candidate=CandidateRecord(spec_hash="parent_cand_123"),
            judge_spec=JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="judge_provider")
            ),
            prompt=PromptTemplateSpec(
                messages=[{"role": "user", "content": "Rate this."}]
            ),
            runtime={
                "dataset_context": {},
                "runtime_context": runtime_context,
            },
        )
    assert [params.seed for params in second_engine.params_seen] == first_run_seeds


def test_default_judge_service_audit_trail_preserves_explicit_judge_seed():
    registry = PluginRegistry()
    engine = MockJudgeInferenceEngine()
    registry.register_inference_engine("judge_provider", engine)
    judge_service = DefaultJudgeService(
        engine_resolver=_resolver_from_registry(registry)
    )

    judge_service.judge(
        metric_id="explicit_seed_metric",
        parent_candidate=CandidateRecord(spec_hash="parent_cand_123"),
        judge_spec=JudgeInferenceSpec(
            model=ModelSpec(model_id="judge-model", provider="judge_provider"),
            params={"seed": 123, "temperature": 0.7},
        ),
        prompt=PromptTemplateSpec(messages=[{"role": "user", "content": "Rate this."}]),
        runtime={
            "dataset_context": {},
            "runtime_context": RuntimeContext(candidate_seed=101),
        },
    )

    trail = judge_service.consume_audit_trail("parent_cand_123")
    assert trail is not None
    assert trail.judge_calls[0].judge_spec.params.seed == 123
    assert trail.judge_calls[0].judge_spec.params.temperature == 0.7


def test_default_judge_service_accepts_explicit_engine_resolver():
    engine = MockJudgeInferenceEngine()
    seen: list[str] = []

    def resolve_engine(provider: str):
        seen.append(provider)
        return engine

    judge_service = DefaultJudgeService(engine_resolver=resolve_engine)

    inf_record = judge_service.judge(
        metric_id="resolver_metric",
        parent_candidate=CandidateRecord(spec_hash="resolver_parent"),
        judge_spec=JudgeInferenceSpec(
            model=ModelSpec(model_id="judge-model", provider="judge_provider")
        ),
        prompt=PromptTemplateSpec(messages=[{"role": "user", "content": "Rate this."}]),
        runtime={"dataset_context": {}},
    )

    assert inf_record.raw_text == "SCORE: 5/5"
    assert seen == ["judge_provider"]
