"""Workflow-runner support types for evaluation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Iterable

from themis.core.contexts import EvalScoreContext
from themis.core.base import JSONValue
from themis.core.events import (
    ProviderCallCompletedEvent,
    ProviderCallFailedEvent,
    ProviderCallStartedEvent,
    RunEvent,
    StepCompletedEvent,
    StepFailedEvent,
    StepStartedEvent,
)
from themis.core.models import MetricResult, TraceStep, WorkflowTrace
from themis.core.planner import Planner
from themis.core.protocols import EvaluationWorkflow, JudgeModel
from themis.core.store import RunStore
from themis.core.subjects import (
    CandidateSetSubject,
    ConversationSubject,
    SessionSubject,
    TraceSubject,
)
from themis.core.workflows import (
    AggregationResult,
    EvaluationExecution,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
    WorkflowFailure,
)


class WorkflowBuildError(ValueError):
    """Raised when a metric cannot build a valid evaluation workflow."""


@dataclass
class _CallExecutionResult:
    response: JudgeResponse | None
    judgment: ParsedJudgment | None
    metric_result: MetricResult | None
    trace_steps: list[TraceStep]
    failure: WorkflowFailure | None = None


class DefaultWorkflowRunner:
    """Concurrent interpreter for Themis-owned evaluation workflows."""

    def __init__(
        self,
        *,
        store: RunStore | None = None,
        judge_models: Iterable[JudgeModel],
        model_call_executor: Callable[
            [JudgeModel, str, int | None], Awaitable[JudgeResponse]
        ]
        | None = None,
        persist_event: Callable[[RunEvent], Awaitable[None]] | None = None,
    ) -> None:
        if store is None and persist_event is None:
            raise ValueError(
                "DefaultWorkflowRunner requires a store or persist_event callback"
            )
        self.store = store
        self.judge_models = {model.component_id: model for model in judge_models}
        self.model_call_executor = model_call_executor
        self.persist_event = persist_event
        self.planner = Planner()

    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject
        | TraceSubject
        | ConversationSubject
        | SessionSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution:
        workflow_id = workflow.component_id
        planned_calls = self.planner.plan_judge_calls(
            run_id=ctx.run_id,
            case_id=ctx.case.case_id,
            case_key=ctx.case_key,
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
                        input={
                            "call_id": call.call_id,
                            "judge_model": call.judge_model_id,
                        },
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

        judge_responses = [
            result.response for result in call_results if result.response is not None
        ]
        parsed_judgments = [
            result.judgment for result in call_results if result.judgment is not None
        ]
        metric_results = [
            result.metric_result
            for result in call_results
            if result.metric_result is not None
        ]
        failures = [
            result.failure for result in call_results if result.failure is not None
        ]
        trace_steps = list(render_trace_steps)
        for result in call_results:
            trace_steps.extend(result.trace_steps)

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
            aggregation_output = workflow.aggregate(
                parsed_judgments, metric_results, ctx
            )
            details: dict[str, JSONValue] = {}
            if aggregation_output is not None:
                details["aggregation_method"] = aggregation_output.method
                details["aggregation_value"] = aggregation_output.value
            trace_steps.append(
                TraceStep(
                    step_name=aggregate_step_id,
                    step_type="aggregate_scores",
                    input={
                        "metric_result_count": len(metric_results),
                        "judgment_count": len(parsed_judgments),
                    },
                    output={
                        "value": aggregation_output.value
                        if aggregation_output is not None
                        else None,
                        "method": aggregation_output.method
                        if aggregation_output is not None
                        else "none",
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
            failures.append(
                WorkflowFailure(
                    step_id=aggregate_step_id,
                    step_type="aggregate_scores",
                    error_message=str(exc),
                )
            )
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=aggregate_step_id,
                    step_type="aggregate_scores",
                    error_message=str(exc),
                )
            )
        status = "partial_failure" if failures else "completed"

        return EvaluationExecution(
            execution_id=f"{ctx.run_id}:{ctx.case.case_id}:{metric_id}:{workflow.fingerprint()}",
            subject_kind=self._subject_kind(subject),
            status=status,
            judge_calls=planned_calls,
            rendered_prompts=rendered_prompts,
            judge_responses=judge_responses,
            parsed_judgments=parsed_judgments,
            metric_results=metric_results,
            failures=failures,
            aggregation_output=aggregation_output,
            trace=WorkflowTrace(
                trace_id=f"{ctx.run_id}:{ctx.case.case_id}:{metric_id}:trace",
                steps=trace_steps,
            ),
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
    ) -> _CallExecutionResult:
        trace_steps: list[TraceStep] = []
        model_step_id = f"{call.call_id}:model_call"
        parse_step_id = f"{call.call_id}:parse_judgment"
        score_step_id = f"{call.call_id}:emit_metric_result"

        await self._persist_event(
            StepStartedEvent(
                run_id=ctx.run_id,
                workflow_id=workflow_id,
                step_id=model_step_id,
                step_type="model_call",
            )
        )
        judge_model: JudgeModel | None = None
        try:
            judge_model = self.judge_models[call.judge_model_id]
            await self._persist_event(
                ProviderCallStartedEvent(
                    run_id=ctx.run_id,
                    case_id=ctx.case.case_id,
                    dataset_id=ctx.dataset_id,
                    case_key=ctx.case_key,
                    stage="judge",
                    provider_id=_provider_id(judge_model),
                    model_id=_provider_model_id(judge_model),
                    provider_key=getattr(judge_model, "provider_key", None),
                    metric_id=metric_id,
                    call_id=call.call_id,
                )
            )
            if self.model_call_executor is None:
                response = await judge_model.judge(
                    prompt.content, seed=call.effective_seed
                )
            else:
                response = await self.model_call_executor(
                    judge_model, prompt.content, call.effective_seed
                )
            response = response.model_copy(
                update={"effective_seed": call.effective_seed}
            )
            await self._persist_event(
                ProviderCallCompletedEvent(
                    run_id=ctx.run_id,
                    case_id=ctx.case.case_id,
                    dataset_id=ctx.dataset_id,
                    case_key=ctx.case_key,
                    stage="judge",
                    provider_id=_provider_id(judge_model),
                    model_id=_provider_model_id(judge_model),
                    provider_key=getattr(judge_model, "provider_key", None),
                    metric_id=metric_id,
                    call_id=call.call_id,
                    telemetry=_judge_provider_telemetry(judge_model, response),
                )
            )
            trace_steps.append(
                TraceStep(
                    step_name=model_step_id,
                    step_type="model_call",
                    input={
                        "prompt": prompt.content,
                        "judge_model": call.judge_model_id,
                        "seed": call.effective_seed,
                    },
                    output={"raw_response": response.raw_response},
                )
            )
            await self._persist_event(
                StepCompletedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=model_step_id,
                    step_type="model_call",
                    details={
                        "judge_model": call.judge_model_id,
                        "effective_seed": call.effective_seed,
                    },
                )
            )
        except Exception as exc:
            retry_history = list(getattr(exc, "retry_history", []))
            await self._persist_event(
                ProviderCallFailedEvent(
                    run_id=ctx.run_id,
                    case_id=ctx.case.case_id,
                    dataset_id=ctx.dataset_id,
                    case_key=ctx.case_key,
                    stage="judge",
                    provider_id=_provider_id(judge_model)
                    if judge_model is not None
                    else call.judge_model_id,
                    model_id=_provider_model_id(judge_model)
                    if judge_model is not None
                    else call.judge_model_id,
                    provider_key=getattr(judge_model, "provider_key", None)
                    if judge_model is not None
                    else None,
                    metric_id=metric_id,
                    call_id=call.call_id,
                    error_message=str(exc),
                    failure_category=_provider_failure_category(exc),
                    retry_history=retry_history,
                )
            )
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=model_step_id,
                    step_type="model_call",
                    error_message=str(exc),
                    retry_history=retry_history,
                )
            )
            return _CallExecutionResult(
                response=None,
                judgment=None,
                metric_result=None,
                trace_steps=trace_steps,
                failure=WorkflowFailure(
                    call_id=call.call_id,
                    step_id=model_step_id,
                    step_type="model_call",
                    error_message=str(exc),
                    retry_history=retry_history,
                ),
            )

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
                    input={
                        "raw_response": response.raw_response,
                        "call_id": call.call_id,
                    },
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
            return _CallExecutionResult(
                response=response,
                judgment=None,
                metric_result=None,
                trace_steps=trace_steps,
                failure=WorkflowFailure(
                    call_id=call.call_id,
                    step_id=parse_step_id,
                    step_type="parse_judgment",
                    error_message=str(exc),
                ),
            )

        await self._persist_event(
            StepStartedEvent(
                run_id=ctx.run_id,
                workflow_id=workflow_id,
                step_id=score_step_id,
                step_type="emit_metric_result",
            )
        )
        try:
            metric_result = workflow.score_judgment(call, judgment, ctx)
            trace_steps.append(
                TraceStep(
                    step_name=score_step_id,
                    step_type="emit_metric_result",
                    input={"label": judgment.label, "call_id": call.call_id},
                    output={
                        "metric_id": metric_id,
                        "value": metric_result.value
                        if metric_result is not None
                        else None,
                    },
                )
            )
            await self._persist_event(
                StepCompletedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=score_step_id,
                    step_type="emit_metric_result",
                    details={
                        "metric_result_emitted": metric_result is not None,
                        "value": metric_result.value
                        if metric_result is not None
                        else None,
                    },
                )
            )
        except Exception as exc:
            await self._persist_event(
                StepFailedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=score_step_id,
                    step_type="emit_metric_result",
                    error_message=str(exc),
                )
            )
            return _CallExecutionResult(
                response=response,
                judgment=judgment,
                metric_result=None,
                trace_steps=trace_steps,
                failure=WorkflowFailure(
                    call_id=call.call_id,
                    step_id=score_step_id,
                    step_type="emit_metric_result",
                    error_message=str(exc),
                ),
            )

        return _CallExecutionResult(
            response=response,
            judgment=judgment,
            metric_result=metric_result,
            trace_steps=trace_steps,
        )

    async def _persist_event(self, event: RunEvent) -> None:
        if self.persist_event is not None:
            await self.persist_event(event)
            return
        if self.store is None:
            raise RuntimeError("No store configured for workflow event persistence")
        self.store.persist_event(event)

    def _subject_kind(
        self,
        subject: CandidateSetSubject
        | TraceSubject
        | ConversationSubject
        | SessionSubject,
    ) -> str:
        if isinstance(subject, CandidateSetSubject):
            return "candidate_set"
        if isinstance(subject, TraceSubject):
            return "trace"
        if isinstance(subject, SessionSubject):
            return "session"
        return "conversation"


def _provider_id(component: object) -> str:
    return str(
        getattr(
            component,
            "provider",
            getattr(component, "component_id", component.__class__.__name__),
        )
    )


def _provider_model_id(component: object) -> str:
    return str(
        getattr(
            component,
            "model_id",
            getattr(component, "component_id", component.__class__.__name__),
        )
    )


def _judge_provider_telemetry(
    component: object, response: JudgeResponse
) -> dict[str, JSONValue]:
    return {
        "provider_id": _provider_id(component),
        "model_id": _provider_model_id(component),
        "latency_ms": float(response.latency_ms or 0.0),
        "retry_count": len(response.retry_history),
        "token_usage": dict(response.token_usage),
        "request_id": response.provider_request_id,
    }


def _provider_failure_category(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return "rate_limit"
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return "provider_unavailable"
    return "provider_failure"
