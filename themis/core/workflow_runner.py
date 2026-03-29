"""Workflow-runner support types for Phase 3 evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from themis.core.contexts import EvalScoreContext
from themis.core.base import JSONValue
from themis.core.events import RunEvent, StepCompletedEvent, StepFailedEvent, StepStartedEvent
from themis.core.models import Score, TraceStep, WorkflowTrace
from themis.core.planner import Planner
from themis.core.protocols import EvaluationWorkflow, JudgeModel
from themis.core.store import RunStore
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject
from themis.core.workflows import (
    AggregationResult,
    EvaluationExecution,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
)


class WorkflowBuildError(ValueError):
    """Raised when a metric cannot build a valid evaluation workflow."""


class DefaultWorkflowRunner:
    """Concurrent interpreter for Themis-owned evaluation workflows."""

    def __init__(
        self,
        *,
        store: RunStore | None = None,
        judge_models: Iterable[JudgeModel],
        model_call_executor: Callable[[JudgeModel, str, int | None], Awaitable[JudgeResponse]] | None = None,
        persist_event: Callable[[RunEvent], Awaitable[None]] | None = None,
    ) -> None:
        if store is None and persist_event is None:
            raise ValueError("DefaultWorkflowRunner requires a store or persist_event callback")
        self.store = store
        self.judge_models = {model.component_id: model for model in judge_models}
        self.model_call_executor = model_call_executor
        self.persist_event = persist_event
        self.planner = Planner()

    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution:
        workflow_id = workflow.component_id
        planned_calls = self.planner.plan_judge_calls(
            run_id=ctx.run_id,
            case_id=ctx.case.case_id,
            metric_id=metric_id,
            calls=workflow.judge_calls(),
        )
        rendered_prompts: list[RenderedJudgePrompt] = []
        render_trace_steps: list[TraceStep] = []
        call_inputs: list[tuple[JudgeCall, RenderedJudgePrompt]] = []

        for call in planned_calls:
            render_step_id = f"{call.call_id}:render_prompt"
            await self._persist_event(
                StepStartedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=render_step_id,
                    step_type="render_prompt",
                )
            )
            try:
                prompt = workflow.render_prompt(call, subject, ctx)
                rendered_prompts.append(prompt)
                call_inputs.append((call, prompt))
                render_trace_steps.append(
                    TraceStep(
                        step_name=render_step_id,
                        step_type="render_prompt",
                        input={"call_id": call.call_id, "judge_model": call.judge_model_id},
                        output={"prompt": prompt.content},
                    )
                )
                await self._persist_event(
                    StepCompletedEvent(
                        run_id=ctx.run_id,
                        workflow_id=workflow_id,
                        step_id=render_step_id,
                        step_type="render_prompt",
                        details={"prompt_id": prompt.prompt_id},
                    )
                )
            except Exception as exc:
                await self._persist_event(
                    StepFailedEvent(
                        run_id=ctx.run_id,
                        workflow_id=workflow_id,
                        step_id=render_step_id,
                        step_type="render_prompt",
                        error_message=str(exc),
                    )
                )
                raise

        call_results = await asyncio.gather(
            *[
                self._execute_call(
                    workflow_id=workflow_id,
                    workflow=workflow,
                    call=call,
                    prompt=prompt,
                    metric_id=metric_id,
                    ctx=ctx,
                )
                for call, prompt in call_inputs
            ]
        )

        judge_responses = [result[0] for result in call_results]
        parsed_judgments = [result[1] for result in call_results]
        scores = [result[2] for result in call_results if result[2] is not None]
        trace_steps = list(render_trace_steps)
        for _, _, _, call_trace_steps in call_results:
            trace_steps.extend(call_trace_steps)

        aggregate_step_id = "aggregate_scores"
        aggregation_output: AggregationResult | None = None
        await self._persist_event(
            StepStartedEvent(
                run_id=ctx.run_id,
                workflow_id=workflow_id,
                step_id=aggregate_step_id,
                step_type="aggregate_scores",
            )
        )
        try:
            aggregation_output = workflow.aggregate(parsed_judgments, scores, ctx)
            details: dict[str, JSONValue] = {}
            if aggregation_output is not None:
                details["aggregation_method"] = aggregation_output.method
                details["aggregation_value"] = aggregation_output.value
            trace_steps.append(
                TraceStep(
                    step_name=aggregate_step_id,
                    step_type="aggregate_scores",
                    input={"score_count": len(scores), "judgment_count": len(parsed_judgments)},
                    output={
                        "value": aggregation_output.value if aggregation_output is not None else None,
                        "method": aggregation_output.method if aggregation_output is not None else "none",
                    },
                )
            )
            await self._persist_event(
                StepCompletedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=aggregate_step_id,
                    step_type="aggregate_scores",
                    details=details,
                )
            )
        except Exception as exc:
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=aggregate_step_id,
                    step_type="aggregate_scores",
                    error_message=str(exc),
                )
            )
            raise

        return EvaluationExecution(
            execution_id=f"{ctx.run_id}:{ctx.case.case_id}:{metric_id}:{workflow.fingerprint()}",
            subject_kind=self._subject_kind(subject),
            judge_calls=planned_calls,
            rendered_prompts=rendered_prompts,
            judge_responses=judge_responses,
            parsed_judgments=parsed_judgments,
            scores=scores,
            aggregation_output=aggregation_output,
            trace=WorkflowTrace(trace_id=f"{ctx.run_id}:{ctx.case.case_id}:{metric_id}:trace", steps=trace_steps),
        )

    async def _execute_call(
        self,
        *,
        workflow_id: str,
        workflow: EvaluationWorkflow,
        call: JudgeCall,
        prompt: RenderedJudgePrompt,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> tuple[JudgeResponse, ParsedJudgment, Score | None, list[TraceStep]]:
        trace_steps: list[TraceStep] = []
        model_step_id = f"{call.call_id}:model_call"
        parse_step_id = f"{call.call_id}:parse_judgment"
        score_step_id = f"{call.call_id}:emit_score"

        await self._persist_event(
            StepStartedEvent(
                run_id=ctx.run_id,
                workflow_id=workflow_id,
                step_id=model_step_id,
                step_type="model_call",
            )
        )
        try:
            judge_model = self.judge_models[call.judge_model_id]
            if self.model_call_executor is None:
                response = await judge_model.judge(prompt.content, seed=call.effective_seed)
            else:
                response = await self.model_call_executor(judge_model, prompt.content, call.effective_seed)
            response = response.model_copy(update={"effective_seed": call.effective_seed})
            trace_steps.append(
                TraceStep(
                    step_name=model_step_id,
                    step_type="model_call",
                    input={"prompt": prompt.content, "judge_model": call.judge_model_id, "seed": call.effective_seed},
                    output={"raw_response": response.raw_response},
                )
            )
            await self._persist_event(
                StepCompletedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=model_step_id,
                    step_type="model_call",
                    details={"judge_model": call.judge_model_id, "effective_seed": call.effective_seed},
                )
            )
        except Exception as exc:
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=model_step_id,
                    step_type="model_call",
                    error_message=str(exc),
                )
            )
            raise

        await self._persist_event(
            StepStartedEvent(
                run_id=ctx.run_id,
                workflow_id=workflow_id,
                step_id=parse_step_id,
                step_type="parse_judgment",
            )
        )
        try:
            judgment = workflow.parse_judgment(call, response, ctx)
            trace_steps.append(
                TraceStep(
                    step_name=parse_step_id,
                    step_type="parse_judgment",
                    input={"raw_response": response.raw_response, "call_id": call.call_id},
                    output={"label": judgment.label, "score": judgment.score},
                )
            )
            await self._persist_event(
                StepCompletedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=parse_step_id,
                    step_type="parse_judgment",
                    details={"label": judgment.label},
                )
            )
        except Exception as exc:
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=parse_step_id,
                    step_type="parse_judgment",
                    error_message=str(exc),
                )
            )
            raise

        await self._persist_event(
            StepStartedEvent(
                run_id=ctx.run_id,
                workflow_id=workflow_id,
                step_id=score_step_id,
                step_type="emit_score",
            )
        )
        try:
            score = workflow.score_judgment(call, judgment, ctx)
            trace_steps.append(
                TraceStep(
                    step_name=score_step_id,
                    step_type="emit_score",
                    input={"label": judgment.label, "call_id": call.call_id},
                    output={"metric_id": metric_id, "value": score.value if score is not None else None},
                )
            )
            await self._persist_event(
                StepCompletedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=score_step_id,
                    step_type="emit_score",
                    details={
                        "score_emitted": score is not None,
                        "score": score.value if score is not None else None,
                    },
                )
            )
        except Exception as exc:
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=score_step_id,
                    step_type="emit_score",
                    error_message=str(exc),
                )
            )
            raise

        return response, judgment, score, trace_steps

    async def _persist_event(self, event: RunEvent) -> None:
        if self.persist_event is not None:
            await self.persist_event(event)
            return
        if self.store is None:
            raise RuntimeError("No store configured for workflow event persistence")
        self.store.persist_event(event)

    def _subject_kind(
        self,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
    ) -> str:
        if isinstance(subject, CandidateSetSubject):
            return "candidate_set"
        if isinstance(subject, TraceSubject):
            return "trace"
        return "conversation"
