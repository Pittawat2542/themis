from __future__ import annotations

import pytest

from themis.core.builtins import resolve_judge_model_component
from themis.core.components import component_ref_from_value
from themis.core.contexts import EvalScoreContext
from themis.core.events import StepCompletedEvent, StepStartedEvent
from themis.core.models import Case, GenerationResult, ParsedOutput
from themis.core.stores.memory import InMemoryRunStore
from themis.core.subjects import CandidateSetSubject
from themis.core.workflow_runner import DefaultWorkflowRunner
from themis.core.workflows import EvalStep, JudgeResponse


class DemoEvaluationWorkflow:
    component_id = "workflow/demo-eval"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-demo-eval-fingerprint"

    def steps(self) -> list[EvalStep]:
        return [
            EvalStep(step_type="render_prompt", config={"template": "Grade candidate: {candidate_output}"}),
            EvalStep(step_type="model_call"),
            EvalStep(step_type="parse_judgment", config={"label_scores": {"pass": 1.0, "fail": 0.0}}),
            EvalStep(step_type="emit_score"),
            EvalStep(step_type="aggregate_scores", config={"method": "mean"}),
        ]


class FixedJudgeModel:
    version = "1.0"

    def __init__(self, component_id: str, response: str) -> None:
        self.component_id = component_id
        self.response = response

    def fingerprint(self) -> str:
        return f"{self.component_id}-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            effective_seed=seed,
            raw_response=self.response if prompt else "fail",
        )


class FanoutWorkflow:
    component_id = "workflow/fanout"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-fanout-fingerprint"

    def steps(self) -> list[EvalStep]:
        return [
            EvalStep(step_type="render_prompt", config={"template": "Helpfulness {candidate_output}"}),
            EvalStep(
                step_type="model_call",
                config={"judge_model": "judge/pass-a", "judge_index": 0, "dimension_id": "helpfulness"},
            ),
            EvalStep(step_type="parse_judgment", config={"label_scores": {"pass": 1.0, "fail": 0.0}}),
            EvalStep(step_type="emit_score"),
            EvalStep(step_type="render_prompt", config={"template": "Accuracy {candidate_output}"}),
            EvalStep(
                step_type="model_call",
                config={"judge_model": "judge/fail", "judge_index": 1, "dimension_id": "accuracy"},
            ),
            EvalStep(step_type="parse_judgment", config={"label_scores": {"pass": 1.0, "fail": 0.0}}),
            EvalStep(step_type="emit_score"),
            EvalStep(step_type="render_prompt", config={"template": "Helpfulness repeat {candidate_output}"}),
            EvalStep(
                step_type="model_call",
                config={
                    "judge_model": "judge/pass-b",
                    "judge_index": 0,
                    "repeat_index": 1,
                    "dimension_id": "helpfulness",
                },
            ),
            EvalStep(step_type="parse_judgment", config={"label_scores": {"pass": 1.0, "fail": 0.0}}),
            EvalStep(step_type="emit_score"),
            EvalStep(step_type="aggregate_scores", config={"method": "majority_vote"}),
        ]


@pytest.mark.asyncio
async def test_default_workflow_runner_executes_single_judge_workflow_and_persists_step_events() -> None:
    store = InMemoryRunStore()
    store.initialize()
    runner = DefaultWorkflowRunner(
        store=store,
        judge_models=[resolve_judge_model_component("judge/demo")],
    )
    subject = CandidateSetSubject(
        candidates=[GenerationResult(candidate_id="candidate-1", final_output={"answer": "4"})]
    )
    ctx = EvalScoreContext(
        run_id="run-1",
        case=Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}),
        parsed_output=ParsedOutput(value={"answer": "4"}),
        judge_model_ref=component_ref_from_value("judge/demo"),
        judge_seed=11,
    )

    execution = await runner.run_evaluation(
        workflow=DemoEvaluationWorkflow(),
        subject=subject,
        metric_id="metric/judge",
        ctx=ctx,
    )

    assert execution.rendered_prompts
    assert execution.judge_responses[0].judge_model_id == "judge/demo"
    assert execution.parsed_judgments
    assert execution.scores
    assert execution.trace.steps

    events = store.query_events("run-1")

    assert [type(event) for event in events] == [
        StepStartedEvent,
        StepCompletedEvent,
        StepStartedEvent,
        StepCompletedEvent,
        StepStartedEvent,
        StepCompletedEvent,
        StepStartedEvent,
        StepCompletedEvent,
        StepStartedEvent,
        StepCompletedEvent,
    ]


@pytest.mark.asyncio
async def test_default_workflow_runner_supports_fanout_and_majority_vote_with_deterministic_seeds() -> None:
    subject = CandidateSetSubject(
        candidates=[GenerationResult(candidate_id="candidate-1", final_output={"answer": "4"})]
    )
    ctx = EvalScoreContext(
        run_id="run-1",
        case=Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}),
        parsed_output=ParsedOutput(value={"answer": "4"}),
        judge_model_ref=component_ref_from_value("judge/demo"),
        judge_seed=11,
    )
    judge_models = [
        FixedJudgeModel("judge/pass-a", "pass"),
        FixedJudgeModel("judge/fail", "fail"),
        FixedJudgeModel("judge/pass-b", "pass"),
    ]

    first_runner = DefaultWorkflowRunner(store=InMemoryRunStore(), judge_models=judge_models)
    second_runner = DefaultWorkflowRunner(store=InMemoryRunStore(), judge_models=judge_models)

    first = await first_runner.run_evaluation(
        workflow=FanoutWorkflow(),
        subject=subject,
        metric_id="metric/judge",
        ctx=ctx,
    )
    second = await second_runner.run_evaluation(
        workflow=FanoutWorkflow(),
        subject=subject,
        metric_id="metric/judge",
        ctx=ctx,
    )

    assert len(first.scores) == 3
    assert first.aggregation_output is not None
    assert first.aggregation_output.method == "majority_vote"
    assert first.aggregation_output.value == 1.0
    assert first.aggregation_output.details["label"] == "pass"
    assert [response.effective_seed for response in first.judge_responses] == [
        response.effective_seed for response in second.judge_responses
    ]
    assert len({response.effective_seed for response in first.judge_responses}) == 3
