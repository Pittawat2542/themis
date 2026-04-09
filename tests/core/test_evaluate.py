from __future__ import annotations

import gc
import warnings
from typing import cast

import pytest

from themis import (
    Experiment,
    InMemoryRunStore,
    RunStatus,
    __version__,
    evaluate,
    evaluate_async,
)
from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Score
from themis.core.models import Case, Dataset
from themis.core.workflows import (
    AggregationResult,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
    build_prompt_template_context,
)
from tests.release import CURRENT_VERSION


class JudgeConfigWorkflow:
    component_id = "workflow/judge-config"
    version = "1.0"

    def __init__(self, metric: "JudgeConfigMetric") -> None:
        self.metric = metric

    def fingerprint(self) -> str:
        return "workflow-judge-config"

    def judge_calls(self) -> list[JudgeCall]:
        return [JudgeCall(call_id="call-1", judge_model_id="builtin/demo_judge")]

    def render_prompt(self, call, subject, ctx) -> RenderedJudgePrompt:
        del call
        template_ctx = build_prompt_template_context(subject, ctx)
        self.metric.seen_template_ctx = template_ctx
        return RenderedJudgePrompt(prompt_id="prompt-1", content="grade")

    def parse_judgment(
        self, call: JudgeCall, response: JudgeResponse, ctx
    ) -> ParsedJudgment:
        del call, ctx
        return ParsedJudgment(label=response.raw_response, score=1.0)

    def score_judgment(
        self, call: JudgeCall, judgment: ParsedJudgment, ctx
    ) -> Score | None:
        del call, ctx
        return Score(
            metric_id="metric/judge-config", value=float(judgment.score or 0.0)
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx,
    ) -> AggregationResult | None:
        del judgments, ctx
        return AggregationResult(
            method="mean",
            value=sum(score.value for score in scores) / len(scores),
        )


class JudgeConfigMetric:
    component_id = "metric/judge-config"
    version = "1.0"
    metric_family = "llm"

    def __init__(self) -> None:
        self.seen_judge_config: dict[str, JSONValue] | None = None
        self.seen_template_ctx: dict[str, JSONValue] | None = None

    def fingerprint(self) -> str:
        return "metric-judge-config"

    def build_workflow(self, subject, ctx):
        del subject
        self.seen_judge_config = dict(ctx.judge_config)
        return JudgeConfigWorkflow(self)


def test_evaluate_runs_one_off_experiment_with_same_run_id_as_explicit_experiment() -> (
    None
):
    datasets = [
        Dataset(
            dataset_id="dataset-1",
            cases=[
                Case(
                    case_id="case-1",
                    input={"question": "2+2"},
                    expected_output={"answer": "4"},
                )
            ],
            revision="r1",
        )
    ]
    explicit = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=datasets,
        seeds=[7],
        themis_version=CURRENT_VERSION,
    )
    store = InMemoryRunStore()

    result = evaluate(
        model="builtin/demo_generator",
        data=datasets,
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
        seeds=[7],
        store=store,
    )

    assert result.status is RunStatus.COMPLETED
    assert result.run_id == explicit.compile().run_id


def test_evaluate_shorthand_supports_workflow_backed_metrics() -> None:
    store = InMemoryRunStore()

    result = evaluate(
        model="builtin/demo_generator",
        data=[
            {
                "case_id": "case-1",
                "input": {"question": "2+2"},
                "expected_output": {"answer": "4"},
            }
        ],
        metric="builtin/llm_rubric",
        parser="builtin/json_identity",
        judge="builtin/demo_judge",
        workflow_overrides={"rubric": "pass if the answer is correct"},
        storage=StorageConfig(store="memory"),
        store=store,
        seeds=[7],
    )

    assert result.status is RunStatus.COMPLETED
    stored = store.resume(result.run_id)

    assert stored is not None
    case_state = next(iter(stored.execution_state.case_states.values()))
    execution = case_state.evaluation_executions["builtin/llm_rubric"]
    assert execution.status == "completed"


def test_evaluate_defaults_release_provenance_to_package_version() -> None:
    store = InMemoryRunStore()

    result = evaluate(
        model="builtin/demo_generator",
        data=[
            {
                "case_id": "case-1",
                "input": {"question": "2+2"},
                "expected_output": {"answer": "4"},
            }
        ],
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
        store=store,
    )

    stored = store.resume(result.run_id)

    assert stored is not None
    assert stored.snapshot.provenance.themis_version == __version__


@pytest.mark.asyncio
async def test_evaluate_async_matches_explicit_experiment_run_async() -> None:
    datasets = [
        Dataset(
            dataset_id="dataset-1",
            cases=[
                Case(
                    case_id="case-1",
                    input={"question": "2+2"},
                    expected_output={"answer": "4"},
                )
            ],
            revision="r1",
        )
    ]
    explicit = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=datasets,
        seeds=[7],
        themis_version=CURRENT_VERSION,
    )

    explicit_result = await explicit.run_async(store=InMemoryRunStore())
    shorthand_result = await evaluate_async(
        model="builtin/demo_generator",
        data=datasets,
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
        seeds=[7],
        store=InMemoryRunStore(),
    )

    assert shorthand_result.status is RunStatus.COMPLETED
    assert shorthand_result.run_id == explicit_result.run_id


@pytest.mark.asyncio
async def test_evaluate_rejects_running_event_loop_with_clear_guidance() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="evaluate_async"):
            evaluate(
                model="builtin/demo_generator",
                data=[
                    {
                        "case_id": "case-1",
                        "input": {"question": "2+2"},
                        "expected_output": {"answer": "4"},
                    }
                ],
                metric="builtin/exact_match",
                parser="builtin/json_identity",
                storage=StorageConfig(store="memory"),
                store=InMemoryRunStore(),
            )
        gc.collect()

    assert not [
        warning for warning in caught if "was never awaited" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_evaluate_async_exposes_judge_config_to_workflows() -> None:
    metric = JudgeConfigMetric()

    result = await evaluate_async(
        model="builtin/demo_generator",
        data=[
            {
                "case_id": "case-1",
                "input": {"question": "2+2"},
                "expected_output": {"answer": "4"},
            }
        ],
        metric=metric,
        parser="builtin/json_identity",
        judge="builtin/demo_judge",
        judge_config={"panel_size": 3},
        storage=StorageConfig(store="memory"),
        store=InMemoryRunStore(),
        seeds=[7],
    )

    assert result.status is RunStatus.COMPLETED
    assert metric.seen_judge_config == {"panel_size": 3}
    assert metric.seen_template_ctx is not None
    assert cast(dict[str, JSONValue], metric.seen_template_ctx["judge_config"]) == {
        "panel_size": 3
    }
