import pytest

from themis.contracts.protocols import InferenceResult
from themis.errors import SpecValidationError
from themis.orchestration.candidate_pipeline import (
    evaluate_candidate,
    generate_candidate,
    transform_candidate,
)
from themis.orchestration.resolved_plugins import resolve_trial_plugins
from themis.orchestration.task_resolution import resolve_task_stages
from themis.records.candidate import CandidateRecord
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.evaluation import MetricScore
from themis.registry.plugin_registry import EngineCapabilities, PluginRegistry
from themis.specs.experiment import (
    InferenceParamsSpec,
    PromptTemplateSpec,
    RuntimeContext,
    TrialSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    JudgeInferenceSpec,
    MetricRefSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.types.enums import DatasetSource, ErrorCode, ErrorWhere, RecordStatus


class MockInferenceEngine:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, trial, context, runtime):
        self.calls += 1
        raw_text = (
            trial.prompt.messages[0].content
            if trial.prompt.messages
            else "The answer is 42."
        )
        return InferenceResult(
            inference=InferenceRecord(spec_hash="inf_hash", raw_text=raw_text)
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


class TrialOnlyMetric:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    def score_trial(self, trial, candidates, context):
        del trial
        self.calls.append(
            (
                len(candidates),
                (
                    context.get("metric_config", {})
                    if isinstance(context.get("metric_config"), dict)
                    else {}
                ).get("strategy"),
            )
        )
        return MetricScore(
            metric_id="trial_metric",
            value=float(len(candidates)),
            details={
                "candidate_ids": [
                    candidate.candidate_id or candidate.spec_hash
                    for candidate in candidates
                ],
                "anchor_candidate_id": context["anchor_candidate"].candidate_id,
            },
        )


class _NoOpHook:
    def pre_inference(self, trial, prompt):
        return prompt

    def post_inference(self, trial, result):
        return result

    def pre_extraction(self, trial, candidate):
        return candidate

    def post_extraction(self, trial, candidate):
        return candidate

    def pre_eval(self, trial, candidate):
        return candidate

    def post_eval(self, trial, candidate):
        return candidate


def _make_trial() -> TrialSpec:
    return TrialSpec(
        trial_id="test_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="t1",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="parsed",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[
                            ExtractorRefSpec(id="mock", config={"needle": "42"})
                        ]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(name="judge", transform="parsed", metrics=["em"])
            ],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )


@pytest.fixture
def registry():
    reg = PluginRegistry()
    engine = MockInferenceEngine()
    reg.register_inference_engine(
        "openai",
        engine,
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


def test_generate_candidate_runs_inference_only(registry):
    trial = _make_trial()

    candidate = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )

    assert candidate.status == RecordStatus.OK
    assert candidate.inference is not None
    assert candidate.inference.raw_text == "The answer is 42."
    assert candidate.extractions == []
    assert candidate.evaluation is None


def test_transform_candidate_reuses_existing_inference(registry):
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    generated = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )

    transformed = transform_candidate(
        trial,
        registry,
        generated,
        resolved.output_transforms[0],
    )

    assert transformed.status == RecordStatus.OK
    assert transformed.inference == generated.inference
    assert len(transformed.extractions) == 1
    assert transformed.extractions[0].success is True
    assert transformed.evaluation is None


def test_transform_candidate_marks_candidate_error_when_all_extractors_fail(registry):
    trial = _make_trial().model_copy(
        update={
            "task": _make_trial().task.model_copy(
                update={
                    "output_transforms": [
                        OutputTransformSpec(
                            name="parsed",
                            extractor_chain=ExtractorChainSpec(
                                extractors=[
                                    ExtractorRefSpec(id="mock", config={"needle": "99"})
                                ]
                            ),
                        )
                    ]
                }
            )
        }
    )
    resolved = resolve_task_stages(trial.task)
    generated = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )

    transformed = transform_candidate(
        trial,
        registry,
        generated,
        resolved.output_transforms[0],
    )

    assert transformed.status == RecordStatus.ERROR
    assert transformed.extractions[-1].success is False
    assert transformed.error is not None
    assert transformed.error.code == ErrorCode.PARSE_ERROR
    assert transformed.error.where == ErrorWhere.EXTRACTOR


def test_evaluate_candidate_reuses_existing_transform_output(registry):
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    generated = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )
    transformed = transform_candidate(
        trial,
        registry,
        generated,
        resolved.output_transforms[0],
    )

    evaluated = evaluate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        transformed,
        resolved.evaluations[0],
    )

    assert evaluated.status == RecordStatus.OK
    assert evaluated.inference == transformed.inference
    assert evaluated.extractions == transformed.extractions
    assert evaluated.evaluation is not None
    assert evaluated.evaluation.aggregate_scores["em"] == 1.0


def test_evaluation_path_does_not_call_inference_again(registry):
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    engine = registry.get_inference_engine("openai")
    generated = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )
    transformed = transform_candidate(
        trial,
        registry,
        generated,
        resolved.output_transforms[0],
    )
    calls_after_generation = engine.calls

    evaluate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        transformed,
        resolved.evaluations[0],
    )

    assert engine.calls == calls_after_generation


def test_stage_functions_apply_hooks_in_order(registry):
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)

    class PromptRewriteHook(_NoOpHook):
        def pre_inference(self, trial, prompt):
            return prompt.model_copy(
                update={"messages": [{"role": "user", "content": "The answer is 43."}]}
            )

    class ExtractionHook(_NoOpHook):
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

    generated = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )
    transformed = transform_candidate(
        trial,
        registry,
        generated,
        resolved.output_transforms[0],
    )
    evaluated = evaluate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        transformed,
        resolved.evaluations[0],
    )

    assert evaluated.status == RecordStatus.OK
    assert evaluated.inference.raw_text == "The answer is 43."
    assert evaluated.extractions[0].warnings == ["normalized"]
    assert evaluated.evaluation.aggregate_scores["em"] == 0.5


def test_stage_functions_can_run_from_resolved_plugins_without_registry_lookups():
    trial = _make_trial()
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor(
        "mock", MockExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric("em", MockMetric, version="1.0.0", plugin_api="1.0")

    class PromptHook(_NoOpHook):
        def pre_inference(self, trial, prompt):
            return prompt.model_copy(
                update={
                    "messages": [
                        {"role": "user", "content": "The answer is 42 (resolved)."}
                    ]
                }
            )

    class EvalHook(_NoOpHook):
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

    registry.register_hook("prompt", PromptHook(), priority=50)
    registry.register_hook("eval", EvalHook(), priority=100)
    resolved_plugins = resolve_trial_plugins(trial, registry)
    empty_registry = PluginRegistry()

    generated = generate_candidate(
        trial,
        empty_registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
        resolved_generation=resolved_plugins.generation,
    )
    transformed = transform_candidate(
        trial,
        empty_registry,
        generated,
        resolved_plugins.resolved_stages.output_transforms[0],
        resolved_transform=resolved_plugins.output_transform_for(
            resolved_plugins.resolved_stages.output_transforms[0].transform_hash
        ),
    )
    evaluated = evaluate_candidate(
        trial,
        empty_registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        transformed,
        resolved_plugins.resolved_stages.evaluations[0],
        resolved_evaluation=resolved_plugins.evaluation_for(
            resolved_plugins.resolved_stages.evaluations[0].evaluation_hash
        ),
    )

    assert generated.inference.raw_text == "The answer is 42 (resolved)."
    assert transformed.extractions[0].success is True
    assert evaluated.evaluation.aggregate_scores["em"] == 0.5


def test_resolved_trial_plugins_are_eager_and_registry_free():
    trial = _make_trial()
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor(
        "mock", MockExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric("em", MockMetric, version="1.0.0", plugin_api="1.0")

    resolved_plugins = resolve_trial_plugins(trial, registry)
    transform_hash = resolved_plugins.resolved_stages.output_transforms[
        0
    ].transform_hash
    evaluation_hash = resolved_plugins.resolved_stages.evaluations[0].evaluation_hash

    assert not hasattr(resolved_plugins, "registry")
    assert (
        resolved_plugins.output_transform_for(transform_hash)
        is resolved_plugins.output_transforms[0]
    )
    assert (
        resolved_plugins.evaluation_for(evaluation_hash)
        is resolved_plugins.evaluations[0]
    )
    assert resolved_plugins.output_transform_for("missing") is None
    assert resolved_plugins.evaluation_for("missing") is None


def test_resolved_trial_plugins_create_judge_service_from_runtime():
    trial = _make_trial()
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_inference_engine("judge_provider", MockInferenceEngine())
    registry.register_extractor(
        "mock", MockExtractor, version="1.0.0", plugin_api="1.0"
    )
    registry.register_metric("em", MockMetric, version="1.0.0", plugin_api="1.0")

    resolved_plugins = resolve_trial_plugins(trial, registry)
    judge_service = resolved_plugins.create_judge_service()

    assert not hasattr(resolved_plugins, "resolve_judge_engine")
    inf_record = judge_service.judge(
        metric_id="judge_metric",
        parent_candidate=CandidateRecord(spec_hash="candidate_1"),
        judge_spec=JudgeInferenceSpec(
            model=ModelSpec(model_id="judge-model", provider="judge_provider")
        ),
        prompt=PromptTemplateSpec(messages=[{"role": "user", "content": "Rate this."}]),
        runtime={"dataset_context": {}},
    )

    assert inf_record.raw_text == "Rate this."


def test_registry_rejects_legacy_two_argument_extractor():
    class LegacyExtractor:
        def extract(self, trial, candidate):
            return ExtractionRecord(
                spec_hash="ext_legacy",
                extractor_id="legacy",
                success=True,
                parsed_answer="42",
            )

    trial = _make_trial().model_copy(
        update={
            "task": _make_trial().task.model_copy(
                update={
                    "output_transforms": [
                        OutputTransformSpec(
                            name="legacy",
                            extractor_chain=ExtractorChainSpec(
                                extractors=[
                                    ExtractorRefSpec(
                                        id="legacy",
                                        config={"needle": "ignored"},
                                    )
                                ]
                            ),
                        )
                    ],
                    "evaluations": [],
                }
            )
        }
    )
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor(
        "legacy",
        LegacyExtractor,
        version="1.0.0",
        plugin_api="1.0",
    )

    with pytest.raises(SpecValidationError, match="extractor"):
        resolve_trial_plugins(trial, registry)


def test_evaluate_candidate_emits_trial_metric_only_on_anchor(registry):
    trial = _make_trial().model_copy(
        update={
            "task": _make_trial().task.model_copy(
                update={
                    "evaluations": [
                        EvaluationSpec(
                            name="judge",
                            transform="parsed",
                            metrics=[
                                MetricRefSpec(
                                    id="trial_metric",
                                    config={"strategy": "majority_vote"},
                                )
                            ],
                        )
                    ]
                }
            )
        }
    )
    trial_metric = TrialOnlyMetric()
    registry.register_metric(
        "trial_metric",
        trial_metric,
        version="1.0.0",
        plugin_api="1.0",
    )
    resolved = resolve_task_stages(trial.task)
    generated_anchor = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=0,
    )
    generated_peer = generate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        cand_index=1,
    )
    transformed_anchor = transform_candidate(
        trial,
        registry,
        generated_anchor,
        resolved.output_transforms[0],
    )
    transformed_peer = transform_candidate(
        trial,
        registry,
        generated_peer,
        resolved.output_transforms[0],
    )

    evaluated_anchor = evaluate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        transformed_anchor,
        resolved.evaluations[0],
        all_candidates=[transformed_anchor, transformed_peer],
    )
    evaluated_peer = evaluate_candidate(
        trial,
        registry,
        {},
        RuntimeContext(environment={"suite": "tests"}),
        transformed_peer,
        resolved.evaluations[0],
        all_candidates=[transformed_anchor, transformed_peer],
    )

    assert evaluated_anchor.evaluation is not None
    assert evaluated_anchor.evaluation.aggregate_scores["trial_metric"] == 2.0
    assert evaluated_anchor.evaluation.metric_scores[0].details[
        "anchor_candidate_id"
    ] == (transformed_anchor.candidate_id)
    assert evaluated_peer.evaluation is not None
    assert evaluated_peer.evaluation.metric_scores == []
    assert trial_metric.calls == [(2, "majority_vote")]
