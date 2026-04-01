from __future__ import annotations

import pytest

from themis.core.builtins import resolve_judge_model_component
from themis.core.components import component_ref_from_value
from themis.core.contexts import EvalScoreContext
from themis.core.events import StepCompletedEvent, StepStartedEvent
from themis.core.models import Case, GenerationResult, ParsedOutput, Score
from themis.core.stores.memory import InMemoryRunStore
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject
from themis.core.workflow_runner import DefaultWorkflowRunner
from themis.core.workflows import (
    AggregationResult,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
)


class DemoEvaluationWorkflow:
    component_id = "workflow/demo-eval"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-demo-eval-fingerprint"

    def judge_calls(self) -> list[JudgeCall]:
        return [JudgeCall(call_id="call-0", judge_model_id="builtin/demo_judge")]

    def render_prompt(
        self,
        call: JudgeCall,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> RenderedJudgePrompt:
        del call, ctx
        assert isinstance(subject, CandidateSetSubject)
        return RenderedJudgePrompt(
            prompt_id="prompt-0",
            content=f"Grade candidate: {subject.candidates[0].final_output}",
        )

    def parse_judgment(
        self,
        call: JudgeCall,
        response: JudgeResponse,
        ctx: EvalScoreContext,
    ) -> ParsedJudgment:
        del call, ctx
        label = response.raw_response.strip().split()[0].lower()
        return ParsedJudgment(
            label=label,
            score=1.0 if label == "pass" else 0.0,
            rationale=response.raw_response,
        )

    def score_judgment(
        self,
        call: JudgeCall,
        judgment: ParsedJudgment,
        ctx: EvalScoreContext,
    ) -> Score | None:
        del call, ctx
        return Score(
            metric_id="metric/judge",
            value=float(judgment.score or 0.0),
            details={"label": judgment.label},
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        return AggregationResult(
            method="mean", value=sum(score.value for score in scores) / len(scores)
        )


class PairwiseSelectionWorkflow:
    component_id = "workflow/pairwise"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-pairwise-fingerprint"

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(
                call_id="call-a-vs-b",
                judge_model_id="judge/selector-a",
                dimension_id="winner",
                candidate_indices=[0, 1],
            ),
            JudgeCall(
                call_id="call-repeat",
                judge_model_id="judge/selector-b",
                dimension_id="winner",
                repeat_index=1,
                candidate_indices=[0, 1],
            ),
        ]

    def render_prompt(
        self,
        call: JudgeCall,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> RenderedJudgePrompt:
        del ctx
        assert isinstance(subject, CandidateSetSubject)
        candidate_a = subject.candidates[call.candidate_indices[0]]
        candidate_b = subject.candidates[call.candidate_indices[1]]
        return RenderedJudgePrompt(
            prompt_id=f"prompt-{call.call_id}",
            content=f"A={candidate_a.final_output}; B={candidate_b.final_output}",
        )

    def parse_judgment(
        self,
        call: JudgeCall,
        response: JudgeResponse,
        ctx: EvalScoreContext,
    ) -> ParsedJudgment:
        del ctx
        return ParsedJudgment(
            label=response.raw_response.strip().lower(),
            details={"call_id": call.call_id},
        )

    def score_judgment(
        self,
        call: JudgeCall,
        judgment: ParsedJudgment,
        ctx: EvalScoreContext,
    ) -> Score | None:
        del ctx
        winner_index = 0 if judgment.label == "a" else 1
        return Score(
            metric_id="metric/select",
            value=float(winner_index),
            details={"call_id": call.call_id, "winner": judgment.label},
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del ctx, scores
        labels = [judgment.label for judgment in judgments]
        winner = max(sorted(set(labels)), key=labels.count)
        return AggregationResult(
            method="majority_vote", value=winner, details={"winner": winner}
        )


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


@pytest.mark.asyncio
async def test_default_workflow_runner_executes_single_judge_workflow_and_persists_step_events() -> (
    None
):
    store = InMemoryRunStore()
    store.initialize()
    runner = DefaultWorkflowRunner(
        store=store,
        judge_models=[resolve_judge_model_component("builtin/demo_judge")],
    )
    subject = CandidateSetSubject(
        candidates=[
            GenerationResult(candidate_id="candidate-1", final_output={"answer": "4"})
        ]
    )
    ctx = EvalScoreContext(
        run_id="run-1",
        case=Case(
            case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
        ),
        parsed_output=ParsedOutput(value={"answer": "4"}),
        judge_model_refs=[component_ref_from_value("builtin/demo_judge")],
        judge_seed=11,
    )

    execution = await runner.run_evaluation(
        workflow=DemoEvaluationWorkflow(),
        subject=subject,
        metric_id="metric/judge",
        ctx=ctx,
    )

    assert execution.rendered_prompts
    assert execution.judge_responses[0].judge_model_id == "builtin/demo_judge"
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
async def test_default_workflow_runner_supports_pairwise_prompts_and_majority_vote_with_deterministic_seeds() -> (
    None
):
    subject = CandidateSetSubject(
        candidates=[
            GenerationResult(candidate_id="candidate-a", final_output={"answer": "4"}),
            GenerationResult(candidate_id="candidate-b", final_output={"answer": "5"}),
        ]
    )
    ctx = EvalScoreContext(
        run_id="run-1",
        case=Case(
            case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
        ),
        parsed_output=ParsedOutput(value={"answer": "4"}),
        judge_model_refs=[
            component_ref_from_value("builtin/demo_judge"),
            component_ref_from_value("builtin/demo_judge"),
        ],
        judge_seed=11,
    )
    judge_models = [
        FixedJudgeModel("judge/selector-a", "a"),
        FixedJudgeModel("judge/selector-b", "a"),
    ]

    first_runner = DefaultWorkflowRunner(
        store=InMemoryRunStore(), judge_models=judge_models
    )
    second_runner = DefaultWorkflowRunner(
        store=InMemoryRunStore(), judge_models=judge_models
    )

    first = await first_runner.run_evaluation(
        workflow=PairwiseSelectionWorkflow(),
        subject=subject,
        metric_id="metric/select",
        ctx=ctx,
    )
    second = await second_runner.run_evaluation(
        workflow=PairwiseSelectionWorkflow(),
        subject=subject,
        metric_id="metric/select",
        ctx=ctx,
    )

    assert [prompt.content for prompt in first.rendered_prompts] == [
        "A={'answer': '4'}; B={'answer': '5'}",
        "A={'answer': '4'}; B={'answer': '5'}",
    ]
    assert len(first.scores) == 2
    assert first.aggregation_output is not None
    assert first.aggregation_output.method == "majority_vote"
    assert first.aggregation_output.value == "a"
    assert first.aggregation_output.details["winner"] == "a"
    assert [response.effective_seed for response in first.judge_responses] == [
        response.effective_seed for response in second.judge_responses
    ]
    assert len({response.effective_seed for response in first.judge_responses}) == 2
