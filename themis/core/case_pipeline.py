"""Per-case stage execution for the Themis runtime."""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeGuard, TypedDict, cast

from themis.core.contexts import (
    ParseContext,
    ReduceContext,
    ScoreContext,
    SelectContext,
)
from themis.core.events import (
    EvaluationCompletedEvent,
    EvaluationFailedEvent,
    ParseCompletedEvent,
    ParseFailedEvent,
    ReductionCompletedEvent,
    ReductionFailedEvent,
    ScoreCompletedEvent,
    ScoreFailedEvent,
    SelectionCompletedEvent,
    SelectionFailedEvent,
)
from themis.core.models import (
    FailureCategory,
    MetricResult,
    ParsedOutput,
    ReducedCandidate,
    ScoreError,
)
from themis.core.protocols import LLMMetric, PureMetric, SelectionMetric, TraceMetric
from themis.core.results import (
    CaseExecutionState,
    CaseResult,
    ExecutionState,
    GenerationWorkItem,
)
from themis.core.snapshot import RunSnapshot
from themis.core.workflow_runner import WorkflowBuildError

WorkflowMetric = LLMMetric | SelectionMetric | TraceMetric
RuntimeMetric = PureMetric | WorkflowMetric


class _CaseIdentityKwargs(TypedDict):
    case_id: str
    dataset_id: str | None
    case_key: str | None


def _is_pure_metric(metric: RuntimeMetric) -> TypeGuard[PureMetric]:
    return isinstance(metric, PureMetric)


def _is_workflow_metric(metric: RuntimeMetric) -> TypeGuard[WorkflowMetric]:
    return (
        isinstance(metric, LLMMetric)
        or isinstance(metric, SelectionMetric)
        or isinstance(metric, TraceMetric)
    )


def _score_pure_metric(
    metric: PureMetric,
    parsed: ParsedOutput,
    case,
    score_ctx: ScoreContext,
) -> MetricResult | ScoreError:
    return metric.score(parsed, case, score_ctx)


class CasePipeline:
    """Executes all stages for one dataset-scoped case."""

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def run_case(
        self,
        snapshot: RunSnapshot,
        items: list[GenerationWorkItem],
        existing_state: ExecutionState,
    ) -> tuple[CaseResult, bool]:
        o = self.orchestrator
        item0 = items[0]
        case = item0.case
        case_result_kwargs: _CaseIdentityKwargs = {
            "case_id": case.case_id,
            "dataset_id": item0.dataset_id,
            "case_key": item0.case_key,
        }
        existing_case_state = existing_state.case_states.get(item0.case_key)
        use_legacy_case_events = False
        if existing_case_state is None:
            existing_case_state = existing_state.case_states.get(case.case_id)
            use_legacy_case_events = existing_case_state is not None
        case_event_kwargs: _CaseIdentityKwargs = {
            "case_id": case.case_id,
            "dataset_id": None if use_legacy_case_events else item0.dataset_id,
            "case_key": None if use_legacy_case_events else item0.case_key,
        }
        prior_case_state = o._replay_case_state(
            existing_case_state or CaseExecutionState()
        )
        prior_case_state = o._rerun_case_state(prior_case_state, case, item0)
        generated_by_index = dict(prior_case_state.generated_candidates_by_index)
        workflow_executions = dict(prior_case_state.evaluation_executions)
        evaluation_failures = dict(prior_case_state.evaluation_failures)
        metric_results = dict(prior_case_state.metric_results)
        score_failures = dict(prior_case_state.score_failures)
        had_failure = False

        pending_generation = [
            o._generate_candidate(snapshot, case, item)
            for item in items
            if item.candidate_index not in generated_by_index
        ]
        for candidate_index, generated, failed in await asyncio.gather(
            *pending_generation
        ):
            if generated is not None:
                generated_by_index[candidate_index] = generated
            had_failure = had_failure or failed

        generated_candidates = [
            generated_by_index[index] for index in sorted(generated_by_index)
        ]
        if not generated_candidates and prior_case_state.reduced_candidate is None:
            return CaseResult(**case_result_kwargs), True
        if o.until_stage == "generate":
            return (
                CaseResult(
                    **case_result_kwargs,
                    generated_candidates=generated_candidates,
                ),
                had_failure,
            )

        selected_candidates = o._selected_candidates_from_state(
            prior_case_state, generated_candidates
        )
        if o.selector is not None and prior_case_state.selected_candidate_ids is None:
            select_ctx = SelectContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                dataset_id=item0.dataset_id,
                case_key=item0.case_key,
                candidate_ids=[
                    candidate.candidate_id for candidate in generated_candidates
                ],
                seed=item0.seed,
                judge_models=list(o.judge_models),
            )
            span = o.tracing_provider.start_span("selection", {"case_id": case.case_id})
            try:
                selected_candidates = await o._select_candidates(
                    generated_candidates, select_ctx
                )
                if not selected_candidates:
                    raise ValueError("Candidate selector returned no candidates")
                await o._persist_event(
                    SelectionCompletedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_ids=[
                            candidate.candidate_id for candidate in selected_candidates
                        ],
                        metadata={"selector_id": o.selector.component_id},
                    )
                )
                o.tracing_provider.end_span(span, "ok")
            except Exception as exc:
                await o._persist_event(
                    SelectionFailedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        error_message=str(exc),
                    )
                )
                o.tracing_provider.end_span(span, "error")
                return CaseResult(
                    **case_result_kwargs,
                    generated_candidates=generated_candidates,
                ), True

        reduced = prior_case_state.reduced_candidate
        if reduced is None:
            cached_reduction = o._load_stage_cache(
                "reduce",
                o._reduction_cache_key(snapshot, selected_candidates),
            )
            if isinstance(cached_reduction, dict) and isinstance(
                cached_reduction.get("result"), dict
            ):
                reduced = ReducedCandidate.model_validate(cached_reduction["result"])
                await o._persist_event(
                    ReductionCompletedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        source_candidate_ids=reduced.source_candidate_ids,
                        result=reduced.model_dump(mode="json"),
                        cache_hit=True,
                        source_run_id=cast(
                            str | None, cached_reduction.get("source_run_id")
                        ),
                    )
                )
        if reduced is None:
            reduce_ctx = ReduceContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                dataset_id=item0.dataset_id,
                case_key=item0.case_key,
                candidate_ids=[
                    candidate.candidate_id for candidate in selected_candidates
                ],
                seed=item0.seed,
                metadata={"selector_id": o.selector.component_id}
                if o.selector is not None
                else {},
            )
            o._notify("before_reduce", selected_candidates, reduce_ctx)
            span = o.tracing_provider.start_span("reduction", {"case_id": case.case_id})
            try:
                reduced = await o._reduce_candidates(selected_candidates, reduce_ctx)
                o._notify("after_reduce", reduced, reduce_ctx)
                await o._persist_event(
                    ReductionCompletedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        source_candidate_ids=reduced.source_candidate_ids,
                        result=reduced.model_dump(mode="json"),
                    )
                )
                o._store_stage_cache(
                    "reduce",
                    o._reduction_cache_key(snapshot, selected_candidates),
                    {
                        "source_run_id": snapshot.run_id,
                        "result": reduced.model_dump(mode="json"),
                    },
                )
                o.tracing_provider.end_span(span, "ok")
            except Exception as exc:
                await o._persist_event(
                    ReductionFailedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        error_message=str(exc),
                    )
                )
                o.tracing_provider.end_span(span, "error")
                return CaseResult(
                    **case_result_kwargs,
                    generated_candidates=generated_candidates,
                ), True
        if o.until_stage == "reduce":
            return (
                CaseResult(
                    **case_result_kwargs,
                    generated_candidates=generated_candidates,
                    reduced_candidate=reduced,
                ),
                had_failure,
            )

        parser_views = list(o.parsers) if o.parsers else [("default", None)]
        parsed_views = dict(prior_case_state.parsed_views)
        parse_errors = dict(prior_case_state.parse_errors)
        for parser_view, parser in parser_views:
            if parser_view in parsed_views:
                continue
            cached_parse = o._load_stage_cache(
                "parse",
                o._parse_cache_key(snapshot, reduced, parser_view),
            )
            if isinstance(cached_parse, dict) and isinstance(
                cached_parse.get("result"), dict
            ):
                parsed = ParsedOutput.model_validate(cached_parse["result"])
                parsed_views[parser_view] = parsed
                parse_errors.pop(parser_view, None)
                await o._persist_event(
                    ParseCompletedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        parser_id=parser_view,
                        result=parsed.model_dump(mode="json"),
                        cache_hit=True,
                        source_run_id=cast(
                            str | None, cached_parse.get("source_run_id")
                        ),
                    )
                )
                continue
            parse_ctx = ParseContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                dataset_id=item0.dataset_id,
                case_key=item0.case_key,
                candidate_id=reduced.candidate_id,
                parser_view=parser_view,
            )
            o._notify("before_parse", reduced, parse_ctx)
            span = o.tracing_provider.start_span(
                "parse", {"case_id": case.case_id, "parser_view": parser_view}
            )
            try:
                async with o._global_semaphore:
                    async with o._stage_semaphores["parsing"]:
                        parsed = await asyncio.to_thread(
                            o._parse_candidate, reduced, parse_ctx, parser
                        )
                parsed_views[parser_view] = parsed
                parse_errors.pop(parser_view, None)
                o._notify("after_parse", parsed, parse_ctx)
                await o._persist_event(
                    ParseCompletedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        parser_id=parser_view,
                        result=parsed.model_dump(mode="json"),
                    )
                )
                o._store_stage_cache(
                    "parse",
                    o._parse_cache_key(snapshot, reduced, parser_view),
                    {
                        "source_run_id": snapshot.run_id,
                        "result": parsed.model_dump(mode="json"),
                    },
                )
                o.tracing_provider.end_span(span, "ok")
            except Exception as exc:
                parse_errors[parser_view] = str(exc)
                await o._persist_event(
                    ParseFailedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        parser_id=parser_view,
                        error_message=str(exc),
                    )
                )
                o.tracing_provider.end_span(span, "error")
                had_failure = True
        if o.until_stage == "parse":
            return (
                CaseResult(
                    **case_result_kwargs,
                    generated_candidates=generated_candidates,
                    reduced_candidate=reduced,
                    parsed_views=parsed_views,
                    parse_errors=parse_errors,
                ),
                had_failure,
            )

        for metric, metric_kind in zip(o.metrics, snapshot.metric_kinds, strict=False):
            if o.until_stage == "score" and metric_kind != "pure":
                continue
            if (
                metric_kind != "pure"
                and metric.component_id not in o.force_workflow_metrics
                and metric.component_id in metric_results
                and metric.component_id in workflow_executions
                and workflow_executions[metric.component_id].status == "completed"
                and not workflow_executions[metric.component_id].failures
            ):
                continue
            if metric_kind == "pure" and metric.component_id in metric_results:
                continue
            parser_view = str(getattr(metric, "parser_view", "default"))
            selected_parsed = parsed_views.get(parser_view)
            if selected_parsed is None:
                score_error = ScoreError(
                    metric_id=metric.component_id,
                    reason=parse_errors.get(
                        parser_view, f"Missing parser view: {parser_view}"
                    ),
                    category=FailureCategory.PARSE_FAILURE
                    if parser_view in parse_errors
                    else FailureCategory.METRIC_FAILURE,
                    metadata={"parser_view": parser_view},
                )
                metric_results.pop(metric.component_id, None)
                score_failures[metric.component_id] = score_error
                await o._persist_event(
                    ScoreFailedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        error=score_error.model_dump(mode="json"),
                    )
                )
                had_failure = True
                continue
            if metric_kind == "pure":
                if not _is_pure_metric(metric):
                    raise TypeError(
                        f"Metric {metric.component_id} does not implement PureMetric"
                    )
                cache_key = o._score_cache_key(snapshot, case, selected_parsed, metric)
                cached_score = o._load_stage_cache("score", cache_key)
                if isinstance(cached_score, dict) and isinstance(
                    cached_score.get("metric_result"), dict
                ):
                    cached_metric_result = MetricResult.model_validate(
                        cached_score["metric_result"]
                    )
                    metric_results[metric.component_id] = cached_metric_result
                    score_failures.pop(metric.component_id, None)
                    await o._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            **case_event_kwargs,
                            candidate_id=reduced.candidate_id,
                            metric_id=cached_metric_result.metric_id,
                            metric_result=cached_metric_result.model_dump(mode="json"),
                            cache_hit=True,
                            source_run_id=cast(
                                str | None, cached_score.get("source_run_id")
                            ),
                        )
                    )
                    continue
                score_ctx = ScoreContext(
                    run_id=snapshot.run_id,
                    case=case,
                    parsed_views=parsed_views,
                    parser_view=parser_view,
                    dataset_id=item0.dataset_id,
                    case_key=item0.case_key,
                    seed=item0.seed,
                )
                o._notify("before_score", selected_parsed, score_ctx)
                span = o.tracing_provider.start_span(
                    "score",
                    {"case_id": case.case_id, "metric_id": metric.component_id},
                )
                try:
                    async with o._stage_semaphores["scoring"]:
                        metric_result = await asyncio.to_thread(
                            _score_pure_metric,
                            metric,
                            selected_parsed,
                            case,
                            score_ctx,
                        )
                    o._notify("after_score", metric_result, score_ctx)
                    if isinstance(metric_result, ScoreError):
                        score_failures[metric.component_id] = metric_result
                        metric_results.pop(metric.component_id, None)
                        await o._persist_event(
                            ScoreFailedEvent(
                                run_id=snapshot.run_id,
                                **case_event_kwargs,
                                candidate_id=reduced.candidate_id,
                                metric_id=metric_result.metric_id,
                                error=metric_result.model_dump(mode="json"),
                            )
                        )
                        had_failure = True
                        o.tracing_provider.end_span(span, "error")
                        continue
                    metric_results[metric.component_id] = metric_result
                    score_failures.pop(metric.component_id, None)
                    await o._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            **case_event_kwargs,
                            candidate_id=reduced.candidate_id,
                            metric_id=metric_result.metric_id,
                            metric_result=metric_result.model_dump(mode="json"),
                        )
                    )
                    o._store_stage_cache(
                        "score",
                        cache_key,
                        {
                            "source_run_id": snapshot.run_id,
                            "metric_result": metric_result.model_dump(mode="json"),
                        },
                    )
                    o.tracing_provider.end_span(span, "ok")
                except Exception as exc:
                    score_error = ScoreError(
                        metric_id=metric.component_id, reason=str(exc)
                    )
                    score_failures[metric.component_id] = score_error
                    metric_results.pop(metric.component_id, None)
                    await o._persist_event(
                        ScoreFailedEvent(
                            run_id=snapshot.run_id,
                            **case_event_kwargs,
                            candidate_id=reduced.candidate_id,
                            metric_id=metric.component_id,
                            error=score_error.model_dump(mode="json"),
                        )
                    )
                    o.tracing_provider.end_span(span, "error")
                    had_failure = True
                continue

            if not _is_workflow_metric(metric):
                raise TypeError(
                    f"Metric {metric.component_id} does not implement a workflow metric protocol"
                )
            eval_ctx = o._evaluation_context(
                snapshot,
                case,
                parsed_views,
                parser_view,
                item0.seed,
                dataset_id=item0.dataset_id,
                case_key=item0.case_key,
            )
            subject = o._evaluation_subject(
                metric_kind=metric_kind,
                generated_candidates=generated_candidates,
                reduced=reduced,
            )
            o._notify("before_judge", subject, eval_ctx)
            span = o.tracing_provider.start_span(
                "judge",
                {"case_id": case.case_id, "metric_id": metric.component_id},
            )
            try:
                workflow = metric.build_workflow(subject, eval_ctx)
                execution = await o.workflow_runner.run_evaluation(
                    workflow=workflow,
                    subject=subject,
                    metric_id=metric.component_id,
                    ctx=eval_ctx,
                )
                o._notify("after_judge", execution, eval_ctx)
                workflow_executions[metric.component_id] = execution
                evaluation_failures.pop(metric.component_id, None)
                execution_blob_ref = await o._store_blob(
                    json.dumps(
                        execution.model_dump(mode="json"), sort_keys=True
                    ).encode("utf-8"),
                    "application/json",
                )
                await o._persist_event(
                    EvaluationCompletedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        execution=execution.model_dump(mode="json"),
                        execution_blob_ref=execution_blob_ref,
                    )
                )
                final_score = o._final_workflow_score(metric.component_id, execution)
                had_failure = (
                    had_failure
                    or execution.status == "partial_failure"
                    or bool(execution.failures)
                )
                if final_score is not None:
                    metric_results[metric.component_id] = final_score
                    score_failures.pop(metric.component_id, None)
                    await o._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            **case_event_kwargs,
                            candidate_id=reduced.candidate_id,
                            metric_id=final_score.metric_id,
                            metric_result=final_score.model_dump(mode="json"),
                        )
                    )
                else:
                    score_error = ScoreError(
                        metric_id=metric.component_id,
                        reason="workflow execution completed without a usable final score",
                    )
                    metric_results.pop(metric.component_id, None)
                    score_failures[metric.component_id] = score_error
                    await o._persist_event(
                        ScoreFailedEvent(
                            run_id=snapshot.run_id,
                            **case_event_kwargs,
                            candidate_id=reduced.candidate_id,
                            metric_id=metric.component_id,
                            error=score_error.model_dump(mode="json"),
                        )
                    )
                    had_failure = True
                o.tracing_provider.end_span(span, "ok")
            except (WorkflowBuildError, Exception) as exc:
                workflow_executions.pop(metric.component_id, None)
                evaluation_failures[metric.component_id] = str(exc)
                score_error = ScoreError(metric_id=metric.component_id, reason=str(exc))
                metric_results.pop(metric.component_id, None)
                score_failures[metric.component_id] = score_error
                await o._persist_event(
                    EvaluationFailedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        error_message=str(exc),
                    )
                )
                await o._persist_event(
                    ScoreFailedEvent(
                        run_id=snapshot.run_id,
                        **case_event_kwargs,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        error=score_error.model_dump(mode="json"),
                    )
                )
                o.tracing_provider.end_span(span, "error")
                had_failure = True

        expected_metric_ids = {
            metric.component_id
            for metric, metric_kind in zip(
                o.metrics, snapshot.metric_kinds, strict=False
            )
            if o.until_stage == "judge" or metric_kind == "pure"
        }

        return (
            CaseResult(
                **case_result_kwargs,
                generated_candidates=generated_candidates,
                reduced_candidate=reduced,
                parsed_views=parsed_views,
                parse_errors=parse_errors,
                evaluation_executions=[
                    workflow_executions[metric.component_id]
                    for metric, metric_kind in zip(
                        o.metrics, snapshot.metric_kinds, strict=False
                    )
                    if metric_kind != "pure"
                    and metric.component_id in workflow_executions
                ],
                metric_results=[
                    result
                    for metric, _metric_kind in zip(
                        o.metrics, snapshot.metric_kinds, strict=False
                    )
                    for result in [
                        metric_results.get(metric.component_id)
                        or score_failures.get(metric.component_id)
                    ]
                    if result is not None
                ],
            ),
            had_failure or len(metric_results) != len(expected_metric_ids),
        )
