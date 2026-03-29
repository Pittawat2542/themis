"""Workflow-runner support types for Phase 3 evaluation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

from themis.core.contexts import EvalScoreContext
from themis.core.events import StepCompletedEvent, StepFailedEvent, StepStartedEvent
from themis.core.models import Score, TraceStep, WorkflowTrace
from themis.core.planner import Planner
from themis.core.protocols import EvaluationWorkflow, JudgeModel
from themis.core.store import RunStore
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject
from themis.core.workflows import (
    AggregationResult,
    EvaluationExecution,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
)


class WorkflowBuildError(ValueError):
    """Raised when a metric cannot build a valid evaluation workflow."""


class DefaultWorkflowRunner:
    """Sequential interpreter for Themis-owned evaluation workflows."""

    def __init__(
        self,
        *,
        store: RunStore,
        judge_models: Iterable[JudgeModel],
        model_call_executor: Callable[[JudgeModel, str, int | None], Awaitable[JudgeResponse]] | None = None,
    ) -> None:
        self.store = store
        self.judge_models = {model.component_id: model for model in judge_models}
        self.model_call_executor = model_call_executor

    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution:
        rendered_prompts: list[RenderedJudgePrompt] = []
        judge_responses: list[JudgeResponse] = []
        parsed_judgments: list[ParsedJudgment] = []
        scores: list[Score] = []
        aggregation_output: AggregationResult | None = None
        trace_steps: list[TraceStep] = []
        workflow_id = workflow.component_id
        subject_kind = self._subject_kind(subject)
        execution_id = f"{ctx.run_id}:{ctx.case.case_id}:{metric_id}:{workflow.fingerprint()}"
        state: dict[str, object] = {}

        for index, step in enumerate(workflow.steps()):
            step_id = f"step-{index}"
            self.store.persist_event(
                StepStartedEvent(
                    run_id=ctx.run_id,
                    workflow_id=workflow_id,
                    step_id=step_id,
                    step_type=step.step_type,
                )
            )
            try:
                details = await self._execute_step(
                    step_id=step_id,
                    step_type=step.step_type,
                    step_config=step.config,
                    subject=subject,
                    metric_id=metric_id,
                    ctx=ctx,
                    rendered_prompts=rendered_prompts,
                    judge_responses=judge_responses,
                    parsed_judgments=parsed_judgments,
                    scores=scores,
                    state=state,
                    trace_steps=trace_steps,
                )
                if step.step_type == "aggregate_scores":
                    aggregation_output = details["aggregation_output"]  # type: ignore[assignment]
                    event_details = {"aggregation_method": aggregation_output.method}
                else:
                    event_details = {
                        key: value
                        for key, value in details.items()
                        if key != "aggregation_output"
                    }
                self.store.persist_event(
                    StepCompletedEvent(
                        run_id=ctx.run_id,
                        workflow_id=workflow_id,
                        step_id=step_id,
                        step_type=step.step_type,
                        details=event_details,
                    )
                )
            except Exception as exc:
                self.store.persist_event(
                    StepFailedEvent(
                        run_id=ctx.run_id,
                        workflow_id=workflow_id,
                        step_id=step_id,
                        step_type=step.step_type,
                        error_message=str(exc),
                    )
                )
                raise

        return EvaluationExecution(
            execution_id=execution_id,
            subject_kind=subject_kind,
            rendered_prompts=rendered_prompts,
            judge_responses=judge_responses,
            parsed_judgments=parsed_judgments,
            scores=scores,
            aggregation_output=aggregation_output,
            trace=WorkflowTrace(trace_id=f"{execution_id}:trace", steps=trace_steps),
        )

    async def _execute_step(
        self,
        *,
        step_id: str,
        step_type: str,
        step_config: dict[str, object],
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        metric_id: str,
        ctx: EvalScoreContext,
        rendered_prompts: list[RenderedJudgePrompt],
        judge_responses: list[JudgeResponse],
        parsed_judgments: list[ParsedJudgment],
        scores: list[Score],
        state: dict[str, object],
        trace_steps: list[TraceStep],
    ) -> dict[str, object]:
        if step_type == "render_prompt":
            prompt = self._render_prompt(subject=subject, ctx=ctx, template=str(step_config.get("template", "{candidate_output}")))
            rendered_prompts.append(RenderedJudgePrompt(prompt_id=step_id, content=prompt))
            state["prompt"] = prompt
            trace_steps.append(
                TraceStep(
                    step_name=step_id,
                    step_type=step_type,
                    input={"template": str(step_config.get("template", "{candidate_output}"))},
                    output={"prompt": prompt},
                )
            )
            return {"prompt_id": step_id}

        if step_type == "model_call":
            prompt = str(state["prompt"])
            judge_model_id = str(step_config.get("judge_model", ctx.judge_model_ref.component_id))
            judge_model = self.judge_models[judge_model_id]
            effective_seed = Planner.judge_seed_for_call(
                run_id=ctx.run_id,
                case_id=ctx.case.case_id,
                metric_id=metric_id,
                judge_index=int(step_config.get("judge_index", 0)),
                repeat_index=int(step_config.get("repeat_index", 0)),
                dimension_id=str(step_config["dimension_id"]) if "dimension_id" in step_config else None,
            )
            if self.model_call_executor is None:
                response = await judge_model.judge(prompt, seed=effective_seed)
            else:
                response = await self.model_call_executor(judge_model, prompt, effective_seed)
            response = response.model_copy(update={"effective_seed": effective_seed})
            judge_responses.append(response)
            state["response"] = response
            trace_steps.append(
                TraceStep(
                    step_name=step_id,
                    step_type=step_type,
                    input={"prompt": prompt, "judge_model": judge_model_id, "seed": effective_seed},
                    output={"raw_response": response.raw_response},
                )
            )
            return {"judge_model": judge_model_id, "effective_seed": effective_seed}

        if step_type == "parse_judgment":
            response = judge_responses[-1]
            label = response.raw_response.strip().split()[0].lower()
            label_scores = step_config.get("label_scores", {})
            score_value = None
            if isinstance(label_scores, dict) and label in label_scores:
                score_value = float(label_scores[label])  # type: ignore[arg-type]
            judgment = ParsedJudgment(label=label, score=score_value, rationale=response.raw_response)
            parsed_judgments.append(judgment)
            state["judgment"] = judgment
            trace_steps.append(
                TraceStep(
                    step_name=step_id,
                    step_type=step_type,
                    input={"raw_response": response.raw_response},
                    output={"label": label, "score": score_value or 0.0},
                )
            )
            return {"label": label}

        if step_type == "emit_score":
            judgment = parsed_judgments[-1]
            score = Score(metric_id=metric_id, value=float(judgment.score or 0.0), details={"label": judgment.label})
            scores.append(score)
            trace_steps.append(
                TraceStep(
                    step_name=step_id,
                    step_type=step_type,
                    input={"label": judgment.label},
                    output={"metric_id": metric_id, "value": score.value},
                )
            )
            return {"score": score.value}

        if step_type == "aggregate_scores":
            method = str(step_config.get("method", "mean"))
            values = [score.value for score in scores]
            if method == "majority_vote":
                labels = [judgment.label for judgment in parsed_judgments]
                winner = ""
                if labels:
                    winner = max(sorted(set(labels)), key=labels.count)
                aggregate_value = 1.0 if winner == "pass" else 0.0
                aggregation_output = AggregationResult(
                    method=method,
                    value=aggregate_value,
                    details={"label": winner},
                )
            else:
                aggregate_value = sum(values) / len(values) if values else 0.0
                aggregation_output = AggregationResult(method=method, value=aggregate_value)
            trace_steps.append(
                TraceStep(
                    step_name=step_id,
                    step_type=step_type,
                    input={"scores": values},
                    output={"value": aggregate_value, "method": method},
                )
            )
            return {"aggregation_output": aggregation_output}

        raise WorkflowBuildError(f"Unsupported workflow step: {step_type}")

    def _render_prompt(
        self,
        *,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
        template: str,
    ) -> str:
        candidate_output = ""
        if isinstance(subject, CandidateSetSubject) and subject.candidates:
            candidate_output = str(subject.candidates[0].final_output)
        elif isinstance(subject, TraceSubject):
            candidate_output = str(subject.trace.model_dump(mode="json"))
        elif isinstance(subject, ConversationSubject):
            candidate_output = str(subject.conversation.model_dump(mode="json"))
        return template.format(
            candidate_output=candidate_output,
            case_input=ctx.case.input,
            parsed_output=ctx.parsed_output.value,
        )

    def _subject_kind(
        self,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
    ) -> str:
        if isinstance(subject, CandidateSetSubject):
            return "candidate_set"
        if isinstance(subject, TraceSubject):
            return "trace"
        return "conversation"
