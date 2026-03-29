from __future__ import annotations

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.planner import Planner
from themis.core.workflows import JudgeCall, JudgeResponse


class DummyLLMMetric:
    component_id = "metric/llm"
    version = "1.0"
    metric_family = "llm"

    def fingerprint(self) -> str:
        return "metric-llm-fingerprint"

    def build_workflow(self, subject, ctx):
        del subject, ctx
        raise NotImplementedError


class DummySelectionMetric:
    component_id = "metric/select"
    version = "1.0"
    metric_family = "selection"

    def fingerprint(self) -> str:
        return "metric-select-fingerprint"

    def build_workflow(self, subject, ctx):
        del subject, ctx
        raise NotImplementedError


class DummyJudgeModel:
    component_id = "judge/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "judge-demo-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response=prompt,
        )


def _experiment(
    *,
    candidate_policy=None,
    reducer="reducer/demo",
    parsers=None,
    metrics=None,
    seeds=None,
    judge_models=None,
):
    return Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy=candidate_policy or {"num_samples": 1},
            reducer=reducer,
        ),
        evaluation=EvaluationConfig(
            metrics=metrics or ["metric/demo"],
            parsers=parsers or ["parser/demo"],
            judge_models=judge_models or [],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}),
                    Case(case_id="case-2", input={"question": "3+3"}, expected_output={"answer": "6"}),
                ],
            )
        ],
        seeds=seeds or [],
    )


@pytest.mark.asyncio
async def test_planner_generates_lazy_candidate_work_items_with_deterministic_seeds() -> None:
    planner = Planner()
    snapshot = _experiment(candidate_policy={"num_samples": 2}).compile()

    first_pass = [item async for item in planner.iter_work_items(snapshot)]
    second_pass = [item async for item in planner.iter_work_items(snapshot)]

    assert [(item.case_id, item.candidate_index, item.seed) for item in first_pass] == [
        ("case-1", 0, first_pass[0].seed),
        ("case-1", 1, first_pass[1].seed),
        ("case-2", 0, first_pass[2].seed),
        ("case-2", 1, first_pass[3].seed),
    ]
    assert [item.seed for item in first_pass] == [item.seed for item in second_pass]
    assert len(first_pass) == 4


@pytest.mark.asyncio
async def test_planner_honors_explicit_seeds_per_candidate() -> None:
    planner = Planner()
    snapshot = _experiment(candidate_policy={"num_samples": 2}, seeds=[7, 11]).compile()

    items = [item async for item in planner.iter_work_items(snapshot)]

    assert [item.seed for item in items[:2]] == [7, 11]
    assert [item.seed for item in items[2:]] == [7, 11]


def test_planner_rejects_multiple_parsers_for_phase_2() -> None:
    planner = Planner()
    snapshot = _experiment(parsers=["parser/demo", "parser/demo"]).compile()

    with pytest.raises(ValueError, match="Phase 2 supports at most one parser"):
        planner.validate_snapshot(snapshot)


def test_planner_requires_reducer_for_multi_candidate_runs() -> None:
    planner = Planner()
    snapshot = _experiment(candidate_policy={"num_samples": 2}, reducer=None, parsers=[]).compile()

    with pytest.raises(ValueError, match="Multi-candidate runs require an explicit reducer"):
        planner.validate_snapshot(snapshot)


def test_planner_allows_workflow_backed_metrics_when_judge_models_are_present() -> None:
    planner = Planner()
    snapshot = _experiment(metrics=[DummyLLMMetric()], judge_models=[DummyJudgeModel()]).compile()

    planner.validate_snapshot(snapshot)


def test_planner_requires_judge_models_for_workflow_backed_metrics() -> None:
    planner = Planner()
    snapshot = _experiment(metrics=[DummyLLMMetric()]).compile()

    with pytest.raises(ValueError, match="judge model"):
        planner.validate_snapshot(snapshot)


def test_planner_requires_multiple_candidates_for_selection_metrics() -> None:
    planner = Planner()
    snapshot = _experiment(metrics=[DummySelectionMetric()], judge_models=[DummyJudgeModel()]).compile()

    with pytest.raises(ValueError, match="Selection metrics require at least two candidates"):
        planner.validate_snapshot(snapshot)


def test_planner_judge_seed_is_deterministic_and_varies_by_fanout_axes() -> None:
    planner = Planner()

    first = planner.judge_seed_for_call(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/llm",
        judge_index=0,
        repeat_index=0,
        dimension_id="helpfulness",
    )
    second = planner.judge_seed_for_call(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/llm",
        judge_index=0,
        repeat_index=0,
        dimension_id="helpfulness",
    )
    different_repeat = planner.judge_seed_for_call(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/llm",
        judge_index=0,
        repeat_index=1,
        dimension_id="helpfulness",
    )

    assert first == second
    assert first != different_repeat


def test_planner_plans_judge_calls_with_effective_seed_and_stable_order() -> None:
    planner = Planner()
    calls = [
        JudgeCall(
            call_id="call-b",
            judge_model_id="judge/demo-b",
            dimension_id="accuracy",
            repeat_index=1,
            candidate_indices=[0, 1],
        ),
        JudgeCall(
            call_id="call-a",
            judge_model_id="judge/demo-a",
            dimension_id="helpfulness",
            candidate_indices=[0],
        ),
    ]

    planned = planner.plan_judge_calls(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/llm",
        calls=calls,
    )

    assert [call.call_id for call in planned] == ["call-a", "call-b"]
    assert planned[0].effective_seed == planner.judge_seed_for_call(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/llm",
        judge_model_id="judge/demo-a",
        repeat_index=0,
        dimension_id="helpfulness",
        candidate_indices=[0],
    )
    assert planned[1].effective_seed == planner.judge_seed_for_call(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/llm",
        judge_model_id="judge/demo-b",
        repeat_index=1,
        dimension_id="accuracy",
        candidate_indices=[0, 1],
    )


def test_planner_requires_self_consistency_count_when_enabled() -> None:
    planner = Planner()
    snapshot = _experiment(candidate_policy={"self_consistency": True}).compile()

    with pytest.raises(ValueError, match="self_consistency_count"):
        planner.validate_snapshot(snapshot)
