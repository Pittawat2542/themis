"""Async execution orchestrator for Themis."""

from __future__ import annotations

import asyncio
import json
from time import monotonic
from collections.abc import Mapping
from typing import Literal, TypeGuard, cast

from themis.core.base import JSONValue
from themis.core.case_pipeline import CasePipeline
from themis.core.config import RuntimeConfig
from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    SelectContext,
    SessionContext,
)
from themis.core.events import (
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    SessionCompletedEvent,
    SessionFailedEvent,
    SessionStartedEvent,
    StreamRecordedEvent,
)
from themis.core.models import (
    ConversationTrace,
    MetricResult,
    ParsedOutput,
    ReducedCandidate,
    SessionResult,
    WorkflowTrace,
)
from themis.core.planner import Planner
from themis.core.projections import build_run_result, build_run_result_from_state
from themis.core.protocols import (
    CandidateReducer,
    CandidateSelector,
    Generator,
    JudgeModel,
    LifecycleSubscriber,
    LLMMetric,
    Parser,
    PureMetric,
    SelectionMetric,
    TraceMetric,
    TracingProvider,
    WorkflowRunner,
)
from themis.core.results import (
    CaseExecutionState,
    CaseResult,
    ExecutionState,
    GenerationWorkItem,
    RerunPlan,
    RunResult,
    RunStatus,
    _case_state_has_failures,
)
from themis.core.runtime_support import (
    RuntimeSupport,
    classify_retryable_error,
    observed_token_cost,
    retry_delay_seconds,
    stable_hash,
)
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.subjects import (
    ConversationSubject,
    SessionSubject,
    TraceSubject,
    candidate_set_subject_for_llm_metric,
    candidate_set_subject_for_selection_metric,
)
from themis.core.tracing import NoOpTracingProvider
from themis.core.workflow_runner import DefaultWorkflowRunner, WorkflowBuildError
from themis.core.workflows import JudgeResponse

WorkflowMetric = LLMMetric | SelectionMetric | TraceMetric
RuntimeMetric = PureMetric | WorkflowMetric
_classify_retryable_error = classify_retryable_error
_retry_delay_seconds = retry_delay_seconds
_observed_token_cost = observed_token_cost
_stable_hash = stable_hash
_compat_monotonic = monotonic


def _is_pure_metric(metric: RuntimeMetric) -> TypeGuard[PureMetric]:
    return isinstance(metric, PureMetric)


def _normalize_parser_views(
    parsers: list[tuple[str, Parser] | tuple[str, Parser, list[Parser]]],
) -> list[tuple[str, Parser, list[Parser]]]:
    normalized: list[tuple[str, Parser, list[Parser]]] = []
    for parser_view in parsers:
        if len(parser_view) == 2:
            view_id, parser = parser_view
            normalized.append((view_id, parser, []))
            continue
        view_id, parser, fallbacks = parser_view
        normalized.append((view_id, parser, list(fallbacks)))
    return normalized


class Orchestrator:
    def __init__(
        self,
        *,
        store: RunStore,
        generator: Generator,
        selector: CandidateSelector | None = None,
        reducer: CandidateReducer | None = None,
        parsers: list[tuple[str, Parser] | tuple[str, Parser, list[Parser]]] | None = None,
        parser: Parser | None = None,
        metrics: list[RuntimeMetric] | None = None,
        judge_models: list[JudgeModel] | None = None,
        workflow_runner: WorkflowRunner | None = None,
        planner: Planner | None = None,
        subscribers: list[LifecycleSubscriber] | None = None,
        tracing_provider: TracingProvider | None = None,
        runtime: RuntimeConfig | None = None,
        max_concurrent_tasks: int | None = None,
        stage_concurrency: dict[str, int] | None = None,
        provider_concurrency: dict[str, int] | None = None,
        provider_limits: dict[str, int] | None = None,
        provider_rate_limits: dict[str, int] | None = None,
        store_retry_delay: float | None = None,
        store_retry_attempts: int | None = None,
        force_workflow_metrics: set[str] | None = None,
        replay_stage: Literal["reduce", "parse", "score", "judge"] | None = None,
        rerun_plan: RerunPlan | None = None,
        until_stage: Literal["generate", "reduce", "parse", "score", "judge"] = "judge",
    ) -> None:
        self.store = store
        self.generator = generator
        self.selector = selector
        self.reducer = reducer
        self.parsers = _normalize_parser_views(
            parsers or ([("default", parser)] if parser else [])
        )
        self.metrics = list(metrics or [])
        self.judge_models = list(judge_models or [])
        self.force_workflow_metrics = set(force_workflow_metrics or set())
        self.replay_stage = replay_stage
        self.rerun_plan = rerun_plan
        self.until_stage = until_stage
        self.planner = planner or Planner()
        self.subscribers = list(subscribers or [])
        self.tracing_provider = tracing_provider or NoOpTracingProvider()
        self.runtime = self._resolve_runtime(
            runtime=runtime,
            max_concurrent_tasks=max_concurrent_tasks,
            stage_concurrency=stage_concurrency,
            provider_concurrency=provider_concurrency or provider_limits,
            provider_rate_limits=provider_rate_limits,
            store_retry_delay=store_retry_delay,
            store_retry_attempts=store_retry_attempts,
        )
        self.runtime_support = RuntimeSupport(
            store=store,
            runtime=self.runtime,
            subscribers=self.subscribers,
            monotonic_clock=monotonic,
        )
        self._global_semaphore = self.runtime_support.global_semaphore
        self._stage_semaphores = self.runtime_support.stage_semaphores
        self.workflow_runner = workflow_runner or DefaultWorkflowRunner(
            store=store,
            judge_models=self.judge_models,
            model_call_executor=self._execute_judge_model_call,
            persist_event=self._persist_event,
        )
        self._case_pipeline = CasePipeline(self)

    async def run(self, snapshot: RunSnapshot) -> RunResult:
        existing_state = self._load_execution_state(snapshot)
        run_span = self.tracing_provider.start_span("run", {"run_id": snapshot.run_id})
        if existing_state.status is RunStatus.PENDING:
            await self._persist_event(RunStartedEvent(run_id=snapshot.run_id))

        case_results: list[CaseResult] = []
        case_failures: list[bool] = []

        try:
            async for case_result, case_failed in self._run_cases(
                snapshot, existing_state
            ):
                case_results.append(case_result)
                case_failures.append(case_failed)
            status = (
                RunStatus.PARTIAL_FAILURE if any(case_failures) else RunStatus.COMPLETED
            )
            await self._persist_event(
                RunCompletedEvent(
                    run_id=snapshot.run_id,
                    completed_through_stage=self.until_stage,
                )
            )
            self.tracing_provider.end_span(
                run_span, "error" if status is RunStatus.PARTIAL_FAILURE else "ok"
            )
            del status, case_results, case_failures
            return self._build_run_result(snapshot)
        except Exception as exc:
            await self._persist_event(
                RunFailedEvent(run_id=snapshot.run_id, error_message=str(exc))
            )
            self.tracing_provider.end_span(run_span, "error")
            raise

    def _load_execution_state(self, snapshot: RunSnapshot) -> ExecutionState:
        checkpoint = self.store.load_execution_checkpoint(snapshot.run_id)
        event_count = self.store.count_events(snapshot.run_id)
        if checkpoint is not None and checkpoint.event_count == event_count:
            return checkpoint.execution_state
        stored_run = self.store.resume(snapshot.run_id)
        if stored_run is None:
            return ExecutionState(run_id=snapshot.run_id)
        return stored_run.execution_state

    def _build_run_result(self, snapshot: RunSnapshot) -> RunResult:
        checkpoint = self.store.load_execution_checkpoint(snapshot.run_id)
        event_count = self.store.count_events(snapshot.run_id)
        if checkpoint is not None and checkpoint.event_count == event_count:
            return build_run_result_from_state(snapshot, checkpoint.execution_state)
        stored_run = self.store.resume(snapshot.run_id)
        if stored_run is None:
            raise RuntimeError(f"Run disappeared from store: {snapshot.run_id}")
        return build_run_result(stored_run.snapshot, stored_run.events)

    async def _run_cases(self, snapshot: RunSnapshot, existing_state: ExecutionState):
        max_in_flight_cases = max(1, self.runtime.max_concurrent_tasks)
        pending: set[asyncio.Task[tuple[CaseResult, bool]]] = set()

        async for items in self._iter_case_groups(snapshot):
            if not items:
                continue
            while len(pending) >= max_in_flight_cases:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    yield task.result()
            pending.add(
                asyncio.create_task(self._run_case(snapshot, items, existing_state))
            )

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                yield task.result()

    async def _iter_case_groups(self, snapshot: RunSnapshot):
        current_case_key: str | None = None
        current_items: list[GenerationWorkItem] = []
        async for item in self.planner.iter_work_items(snapshot):
            if current_case_key is None:
                current_case_key = item.case_key
            if item.case_key != current_case_key:
                yield current_items
                current_items = []
                current_case_key = item.case_key
            current_items.append(item)
        if current_items:
            yield current_items

    async def _run_case(
        self,
        snapshot: RunSnapshot,
        items: list[GenerationWorkItem],
        existing_state: ExecutionState,
    ) -> tuple[CaseResult, bool]:
        return await self._case_pipeline.run_case(snapshot, items, existing_state)

    async def _generate_candidate(
        self,
        snapshot: RunSnapshot,
        case,
        item: GenerationWorkItem,
    ) -> tuple[int, SessionResult | None, bool]:
        termination = snapshot.identity.candidate_policy.get("termination")
        max_turns = snapshot.identity.candidate_policy.get("max_turns", 1)
        if not isinstance(max_turns, int):
            max_turns = 1
        session_ctx = SessionContext(
            run_id=snapshot.run_id,
            case_id=item.case_id,
            dataset_id=item.dataset_id,
            case_key=item.case_key,
            seed=item.seed,
            prompt_spec=snapshot.identity.generation_prompt_spec,
            max_turns=max_turns,
            termination=termination if isinstance(termination, dict) else {},
        )
        cache_key = self._generation_cache_key(snapshot, case, item)
        cached_generation = self._load_stage_cache("generate", cache_key)
        if isinstance(cached_generation, dict) and isinstance(
            cached_generation.get("result"), dict
        ):
            generated = SessionResult.model_validate(cached_generation["result"])
            blob_ref = await self._store_blob(
                json.dumps(generated.model_dump(mode="json"), sort_keys=True).encode(
                    "utf-8"
                ),
                "application/json",
            )
            for stream_event in generated.stream_events:
                await self._persist_event(
                    StreamRecordedEvent(
                        run_id=snapshot.run_id,
                        case_id=item.case_id,
                        dataset_id=item.dataset_id,
                        case_key=item.case_key,
                        candidate_id=generated.candidate_id,
                        source_stage=stream_event.source_stage,
                        stream_event=stream_event.model_dump(mode="json"),
                    )
                )
            await self._persist_event(
                SessionCompletedEvent(
                    run_id=snapshot.run_id,
                    case_id=item.case_id,
                    dataset_id=item.dataset_id,
                    case_key=item.case_key,
                    candidate_id=generated.candidate_id,
                    candidate_index=item.candidate_index,
                    seed=item.seed,
                    provider_key=cast(
                        str | None, cached_generation.get("provider_key")
                    ),
                    result=generated.model_dump(mode="json"),
                    result_blob_ref=blob_ref,
                    cache_hit=True,
                    source_run_id=cast(
                        str | None, cached_generation.get("source_run_id")
                    ),
                )
            )
            return item.candidate_index, generated, False
        async with self._global_semaphore:
            async with self._stage_semaphores["generation"]:
                provider_key = self._provider_key()
                provider_semaphore = (
                    self._provider_semaphore(provider_key)
                    if provider_key is not None
                    else None
                )
                provider_limiter = (
                    self._provider_limiter(provider_key)
                    if provider_key is not None
                    else None
                )
                provider_token_limiter = (
                    self._provider_token_limiter(provider_key)
                    if provider_key is not None
                    else None
                )
                if provider_semaphore is not None:
                    await provider_semaphore.acquire()
                if provider_limiter is not None:
                    await provider_limiter.acquire()
                if provider_token_limiter is not None:
                    await provider_token_limiter.acquire()
                try:
                    self._notify("before_generate", case, session_ctx)
                    span = self.tracing_provider.start_span(
                        "generation", {"case_id": item.case_id}
                    )
                    try:
                        await self._persist_event(
                            SessionStartedEvent(
                                run_id=snapshot.run_id,
                                case_id=item.case_id,
                                dataset_id=item.dataset_id,
                                case_key=item.case_key,
                                candidate_id=item.candidate_id,
                                candidate_index=item.candidate_index,
                                seed=item.seed,
                                provider_key=provider_key,
                            )
                        )
                        generated = await self._run_session_with_retries(
                            case, session_ctx
                        )
                        if provider_token_limiter is not None:
                            await provider_token_limiter.acquire(
                                _observed_token_cost(generated.token_usage) - 1
                            )
                        await self._update_rate_limit(provider_key, generated.artifacts)
                        self._notify("after_generate", generated, session_ctx)
                        blob_ref = await self._store_blob(
                            json.dumps(
                                generated.model_dump(mode="json"), sort_keys=True
                            ).encode("utf-8"),
                            "application/json",
                        )
                        for stream_event in generated.stream_events:
                            await self._persist_event(
                                StreamRecordedEvent(
                                    run_id=snapshot.run_id,
                                    case_id=item.case_id,
                                    dataset_id=item.dataset_id,
                                    case_key=item.case_key,
                                    candidate_id=generated.candidate_id,
                                    source_stage=stream_event.source_stage,
                                    stream_event=stream_event.model_dump(mode="json"),
                                )
                            )
                        await self._persist_event(
                            SessionCompletedEvent(
                                run_id=snapshot.run_id,
                                case_id=item.case_id,
                                dataset_id=item.dataset_id,
                                case_key=item.case_key,
                                candidate_id=generated.candidate_id,
                                candidate_index=item.candidate_index,
                                seed=item.seed,
                                provider_key=provider_key,
                                result=generated.model_dump(mode="json"),
                                result_blob_ref=blob_ref,
                            )
                        )
                        self._store_stage_cache(
                            "generate",
                            cache_key,
                            {
                                "source_run_id": snapshot.run_id,
                                "provider_key": provider_key,
                                "result": generated.model_dump(mode="json"),
                            },
                        )
                        self.tracing_provider.end_span(span, "ok")
                        return item.candidate_index, generated, False
                    except Exception as exc:
                        retry_history = getattr(exc, "retry_history", [])
                        await self._persist_event(
                            SessionFailedEvent(
                                run_id=snapshot.run_id,
                                case_id=item.case_id,
                                dataset_id=item.dataset_id,
                                case_key=item.case_key,
                                candidate_id=item.candidate_id,
                                candidate_index=item.candidate_index,
                                error_message=str(exc),
                                retry_history=retry_history,
                            )
                        )
                        self.tracing_provider.end_span(span, "error")
                        return item.candidate_index, None, True
                finally:
                    if provider_semaphore is not None:
                        provider_semaphore.release()

    async def _select_candidates(
        self,
        generated_candidates: list[SessionResult],
        select_ctx: SelectContext,
    ) -> list[SessionResult]:
        if self.selector is None:
            return generated_candidates
        async with self._global_semaphore:
            async with self._stage_semaphores["selection"]:
                return await self.selector.select(generated_candidates, select_ctx)

    async def _reduce_candidates(
        self,
        generated_candidates: list[SessionResult],
        reduce_ctx: ReduceContext,
    ) -> ReducedCandidate:
        async with self._global_semaphore:
            async with self._stage_semaphores["reduction"]:
                if self.reducer is None:
                    candidate = generated_candidates[0]
                    return ReducedCandidate(
                        candidate_id=candidate.candidate_id,
                        source_candidate_ids=[
                            candidate.candidate_id for candidate in generated_candidates
                        ],
                        final_output=candidate.final_output,
                    )
                return await self.reducer.reduce(generated_candidates, reduce_ctx)

    def _parse_candidate(
        self,
        reduced: ReducedCandidate,
        parse_ctx: ParseContext,
        parser: Parser | None,
    ) -> ParsedOutput:
        if parser is None:
            return ParsedOutput(value=reduced.final_output)
        return parser.parse(reduced, parse_ctx)

    def _evaluation_context(
        self,
        snapshot: RunSnapshot,
        case,
        parsed_views: dict[str, ParsedOutput],
        parser_view: str,
        seed: int | None,
        *,
        dataset_id: str | None,
        case_key: str | None,
    ) -> EvalScoreContext:
        if not snapshot.identity.judge_model_refs:
            raise WorkflowBuildError(
                "Workflow-backed metrics require at least one judge model"
            )
        return EvalScoreContext(
            run_id=snapshot.run_id,
            case=case,
            parsed_views=parsed_views,
            parser_view=parser_view,
            dataset_id=dataset_id,
            case_key=case_key,
            seed=seed,
            judge_model_refs=list(snapshot.identity.judge_model_refs),
            judge_seed=seed,
            prompt_spec=snapshot.identity.evaluation_prompt_spec,
            judge_config=snapshot.identity.judge_config,
            eval_workflow_config=snapshot.identity.workflow_overrides,
        )

    def _evaluation_subject(
        self,
        *,
        metric_kind: str,
        generated_candidates: list[SessionResult],
        reduced: ReducedCandidate,
    ):
        if metric_kind == "llm":
            reduced_candidate = SessionResult(
                candidate_id=reduced.candidate_id,
                final_output=reduced.final_output,
            )
            return candidate_set_subject_for_llm_metric([reduced_candidate])
        if metric_kind == "selection":
            return candidate_set_subject_for_selection_metric(generated_candidates)
        if metric_kind == "trace":
            winner_id = (
                reduced.source_candidate_ids[0]
                if reduced.source_candidate_ids
                else generated_candidates[0].candidate_id
            )
            winner = next(
                (
                    candidate
                    for candidate in generated_candidates
                    if candidate.candidate_id == winner_id
                ),
                generated_candidates[0],
            )
            if winner.trace is not None:
                return TraceSubject(
                    trace=WorkflowTrace(
                        trace_id=f"{winner.candidate_id}:trace", steps=winner.trace
                    )
                )
            if winner.conversation is not None:
                return ConversationSubject(
                    conversation=ConversationTrace(
                        trace_id=f"{winner.candidate_id}:conversation",
                        messages=winner.conversation,
                    )
                )
            if winner.turns or winner.stream_events:
                return SessionSubject(session=winner)
            raise WorkflowBuildError(
                "Trace metrics require a trace, conversation, or session artifact on the winning candidate"
            )
        raise WorkflowBuildError(f"Unsupported metric kind: {metric_kind}")

    def _final_workflow_score(
        self,
        metric_id: str,
        execution,
    ) -> MetricResult | None:
        if execution.aggregation_output is not None:
            if execution.aggregation_output.metric_result is not None:
                return execution.aggregation_output.metric_result
            if isinstance(execution.aggregation_output.value, (int, float)):
                return MetricResult(
                    metric_id=metric_id,
                    value=float(execution.aggregation_output.value),
                    metadata=execution.aggregation_output.metadata,
                )
        if not execution.failures and len(execution.metric_results) == 1:
            return execution.metric_results[-1]
        return None

    async def _execute_judge_model_call(
        self,
        judge_model: JudgeModel,
        prompt: str,
        seed: int | None,
    ) -> JudgeResponse:
        async with self._global_semaphore:
            async with self._stage_semaphores["evaluation"]:
                provider_key = getattr(judge_model, "provider_key", None)
                provider_semaphore = (
                    self._provider_semaphore(provider_key)
                    if provider_key is not None
                    else None
                )
                provider_limiter = (
                    self._provider_limiter(provider_key)
                    if provider_key is not None
                    else None
                )
                provider_token_limiter = (
                    self._provider_token_limiter(provider_key)
                    if provider_key is not None
                    else None
                )
                if provider_semaphore is not None:
                    await provider_semaphore.acquire()
                if provider_limiter is not None:
                    await provider_limiter.acquire()
                if provider_token_limiter is not None:
                    await provider_token_limiter.acquire()
                try:
                    response = await self._judge_with_retries(
                        judge_model, prompt, seed=seed
                    )
                    if provider_token_limiter is not None:
                        await provider_token_limiter.acquire(
                            _observed_token_cost(response.token_usage) - 1
                        )
                    return response
                finally:
                    if provider_semaphore is not None:
                        provider_semaphore.release()

    def _replay_case_state(self, case_state: CaseExecutionState) -> CaseExecutionState:
        if self.replay_stage is None:
            return case_state

        workflow_metric_ids = {
            metric.component_id
            for metric, metric_kind in zip(
                self.metrics, self._metric_kinds(), strict=False
            )
            if metric_kind != "pure"
        }
        metric_results = dict(case_state.metric_results)
        score_failures = dict(case_state.score_failures)

        if self.replay_stage == "judge":
            for metric_id in workflow_metric_ids:
                metric_results.pop(metric_id, None)
                score_failures.pop(metric_id, None)
            return case_state.model_copy(
                update={
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "metric_results": metric_results,
                    "score_failures": score_failures,
                }
            )

        if self.replay_stage == "score":
            return case_state.model_copy(
                update={
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "metric_results": {},
                    "score_failures": {},
                }
            )

        if self.replay_stage == "parse":
            return case_state.model_copy(
                update={
                    "parsed_views": {},
                    "parse_errors": {},
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "metric_results": {},
                    "score_failures": {},
                }
            )

        return case_state.model_copy(
            update={
                "reduced_candidate": None,
                "reduction_error": None,
                "parsed_views": {},
                "parse_errors": {},
                "evaluation_executions": {},
                "evaluation_execution_blob_refs": {},
                "evaluation_failures": {},
                "metric_results": {},
                "score_failures": {},
            }
        )

    def _rerun_case_state(
        self,
        case_state: CaseExecutionState,
        case,
        item: GenerationWorkItem,
    ) -> CaseExecutionState:
        if self.rerun_plan is None:
            return case_state
        if not self._case_matches_rerun_plan(case_state, case, item):
            return case_state

        metric_ids = set(self.rerun_plan.selector.metric_ids)
        stage = self.rerun_plan.stage
        if stage == "generate":
            return CaseExecutionState()
        if stage == "reduce":
            return case_state.model_copy(
                update={
                    "selected_candidate_ids": None,
                    "selection_metadata": {},
                    "selection_error": None,
                    "reduced_candidate": None,
                    "reduction_error": None,
                    "parsed_views": {},
                    "parse_errors": {},
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "metric_results": {},
                    "score_failures": {},
                }
            )
        if stage == "parse":
            return case_state.model_copy(
                update={
                    "parsed_views": {},
                    "parse_errors": {},
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "metric_results": {},
                    "score_failures": {},
                }
            )

        metric_results = dict(case_state.metric_results)
        score_failures = dict(case_state.score_failures)
        evaluation_executions = dict(case_state.evaluation_executions)
        evaluation_execution_blob_refs = dict(case_state.evaluation_execution_blob_refs)
        evaluation_failures = dict(case_state.evaluation_failures)
        score_ids_to_clear = metric_ids or set(metric_results) | set(score_failures)
        for metric_id in score_ids_to_clear:
            metric_results.pop(metric_id, None)
            score_failures.pop(metric_id, None)

        if stage == "judge":
            workflow_metric_ids = metric_ids or {
                metric.component_id
                for metric, metric_kind in zip(
                    self.metrics, self._metric_kinds(), strict=False
                )
                if metric_kind != "pure"
            }
            for metric_id in workflow_metric_ids:
                evaluation_executions.pop(metric_id, None)
                evaluation_execution_blob_refs.pop(metric_id, None)
                evaluation_failures.pop(metric_id, None)
                metric_results.pop(metric_id, None)
                score_failures.pop(metric_id, None)

        return case_state.model_copy(
            update={
                "evaluation_executions": evaluation_executions,
                "evaluation_execution_blob_refs": evaluation_execution_blob_refs,
                "evaluation_failures": evaluation_failures,
                "metric_results": metric_results,
                "score_failures": score_failures,
            }
        )

    def _case_matches_rerun_plan(
        self,
        case_state: CaseExecutionState,
        case,
        item: GenerationWorkItem,
    ) -> bool:
        if self.rerun_plan is None:
            return False
        selector = self.rerun_plan.selector
        if selector.failed_only and not _case_state_has_failures(case_state):
            return False
        if selector.case_ids and case.case_id not in selector.case_ids:
            return False
        if selector.case_keys and item.case_key not in selector.case_keys:
            return False
        if selector.metadata and any(
            case.metadata.get(key) != value for key, value in selector.metadata.items()
        ):
            return False
        return True

    def _metric_kinds(self) -> list[str]:
        return [
            "pure" if _is_pure_metric(metric) else "workflow" for metric in self.metrics
        ]

    def _selected_candidates_from_state(
        self,
        case_state: CaseExecutionState,
        generated_candidates: list[SessionResult],
    ) -> list[SessionResult]:
        if case_state.selected_candidate_ids is None:
            return generated_candidates
        selected_by_id = {
            candidate.candidate_id: candidate for candidate in generated_candidates
        }
        return [
            selected_by_id[candidate_id]
            for candidate_id in case_state.selected_candidate_ids
            if candidate_id in selected_by_id
        ]

    async def _run_session_with_retries(
        self, case, session_ctx: SessionContext
    ) -> SessionResult:
        retry_history: list[dict[str, JSONValue]] = []
        for attempt in range(self.runtime.generation_retry_attempts):
            try:
                if hasattr(self.generator, "run_session"):
                    generated = await self.generator.run_session(case, session_ctx)
                else:
                    legacy_ctx = GenerateContext.model_validate(
                        session_ctx.model_dump(mode="json")
                    )
                    generated = await self.generator.generate(case, legacy_ctx)
                session_result = SessionResult.model_validate(
                    generated.model_dump(mode="json")
                    if hasattr(generated, "model_dump")
                    else generated
                )
                if retry_history:
                    artifacts = dict(session_result.artifacts or {})
                    artifacts["retry_history"] = cast(JSONValue, retry_history)
                    session_result = session_result.model_copy(
                        update={"artifacts": artifacts}
                    )
                return session_result
            except Exception as exc:
                retry_classification = _classify_retryable_error(exc)
                if attempt + 1 == self.runtime.generation_retry_attempts or (
                    retry_classification is None
                ):
                    setattr(exc, "retry_history", retry_history)
                    raise
                delay = _retry_delay_seconds(
                    base_delay=self.runtime.generation_retry_delay,
                    backoff=self.runtime.generation_retry_backoff,
                    attempt=attempt,
                    retry_after_s=retry_classification.get("retry_after_s"),
                )
                retry_entry: dict[str, JSONValue] = {
                    "attempt": attempt + 1,
                    "error_message": str(exc),
                    "delay_s": delay,
                    "reason": retry_classification["reason"],
                }
                retry_after_s = retry_classification.get("retry_after_s")
                if retry_after_s is not None:
                    retry_entry["retry_after_s"] = retry_after_s
                retry_history.append(retry_entry)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    async def _judge_with_retries(
        self,
        judge_model: JudgeModel,
        prompt: str,
        *,
        seed: int | None,
    ) -> JudgeResponse:
        retry_history: list[dict[str, JSONValue]] = []
        for attempt in range(self.runtime.judge_retry_attempts):
            try:
                response = await judge_model.judge(prompt, seed=seed)
                if retry_history:
                    response = response.model_copy(
                        update={"retry_history": retry_history}
                    )
                return response
            except Exception as exc:
                retry_classification = _classify_retryable_error(exc)
                if attempt + 1 == self.runtime.judge_retry_attempts or (
                    retry_classification is None
                ):
                    setattr(exc, "retry_history", retry_history)
                    raise
                delay = _retry_delay_seconds(
                    base_delay=self.runtime.judge_retry_delay,
                    backoff=self.runtime.judge_retry_backoff,
                    attempt=attempt,
                    retry_after_s=retry_classification.get("retry_after_s"),
                )
                retry_entry: dict[str, JSONValue] = {
                    "attempt": attempt + 1,
                    "error_message": str(exc),
                    "delay_s": delay,
                    "reason": retry_classification["reason"],
                }
                retry_after_s = retry_classification.get("retry_after_s")
                if retry_after_s is not None:
                    retry_entry["retry_after_s"] = retry_after_s
                retry_history.append(retry_entry)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    async def _persist_event(self, event) -> None:
        await self.runtime_support.persist_event(event)

    async def _store_blob(self, blob: bytes, media_type: str) -> str:
        return await self.runtime_support.store_blob(blob, media_type)

    def _notify(self, method_name: str, *args) -> None:
        self.runtime_support.notify(method_name, *args)

    def _provider_key(self) -> str | None:
        return getattr(self.generator, "provider_key", None)

    def _provider_semaphore(self, provider_key: str) -> asyncio.Semaphore:
        return self.runtime_support.provider_semaphore(provider_key)

    def _provider_limiter(self, provider_key: str):
        return self.runtime_support.provider_limiter(provider_key)

    def _provider_token_limiter(self, provider_key: str):
        return self.runtime_support.provider_token_limiter(provider_key)

    async def _update_rate_limit(
        self,
        provider_key: str | None,
        artifacts: Mapping[str, object] | None,
    ) -> None:
        await self.runtime_support.update_rate_limit(provider_key, artifacts)

    def _load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        return self.runtime_support.load_stage_cache(stage_name, cache_key)

    def _store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        self.runtime_support.store_stage_cache(stage_name, cache_key, payload)

    def _generation_cache_key(
        self, snapshot: RunSnapshot, case, item: GenerationWorkItem
    ) -> str:
        return _stable_hash(
            {
                "stage": "generate",
                "case_hash": case.compute_hash(),
                "generator_ref": snapshot.component_refs.generator.model_dump(
                    mode="json"
                ),
                "candidate_policy": snapshot.identity.candidate_policy,
                "prompt_spec": snapshot.identity.generation_prompt_spec.model_dump(
                    mode="json"
                )
                if snapshot.identity.generation_prompt_spec is not None
                else None,
                "seed": item.seed,
            }
        )

    def _reduction_cache_key(
        self, snapshot: RunSnapshot, candidates: list[SessionResult]
    ) -> str:
        return _stable_hash(
            {
                "stage": "reduce",
                "selector_ref": snapshot.component_refs.selector.model_dump(mode="json")
                if snapshot.component_refs.selector is not None
                else None,
                "reducer_ref": snapshot.component_refs.reducer.model_dump(mode="json")
                if snapshot.component_refs.reducer is not None
                else None,
                "candidate_hashes": [
                    candidate.compute_hash() for candidate in candidates
                ],
            }
        )

    def _parse_cache_key(
        self, snapshot: RunSnapshot, reduced: ReducedCandidate, parser_view: str
    ) -> str:
        parser_view_ref = next(
            (
                view_ref
                for view_ref in snapshot.component_refs.parsers
                if view_ref.id == parser_view
            ),
            None,
        )
        return _stable_hash(
            {
                "stage": "parse",
                "parser_view": parser_view,
                "reduced_hash": reduced.compute_hash(),
                "parser_ref": parser_view_ref.parser.model_dump(mode="json")
                if parser_view_ref is not None
                else None,
                "fallback_refs": [
                    fallback.model_dump(mode="json")
                    for fallback in parser_view_ref.fallbacks
                ]
                if parser_view_ref is not None
                else [],
            }
        )

    def _score_cache_key(
        self, snapshot: RunSnapshot, case, parsed: ParsedOutput, metric: PureMetric
    ) -> str:
        metric_index = next(
            index
            for index, candidate_metric in enumerate(self.metrics)
            if candidate_metric.component_id == metric.component_id
        )
        return _stable_hash(
            {
                "stage": "score",
                "case_hash": case.compute_hash(),
                "parsed_hash": parsed.compute_hash(),
                "metric_ref": snapshot.component_refs.metrics[metric_index].model_dump(
                    mode="json"
                ),
                "prompt_spec": snapshot.identity.evaluation_prompt_spec.model_dump(
                    mode="json"
                )
                if snapshot.identity.evaluation_prompt_spec is not None
                else None,
            }
        )

    def _resolve_runtime(
        self,
        *,
        runtime: RuntimeConfig | None,
        max_concurrent_tasks: int | None,
        stage_concurrency: dict[str, int] | None,
        provider_concurrency: dict[str, int] | None,
        provider_rate_limits: dict[str, int] | None,
        store_retry_delay: float | None,
        store_retry_attempts: int | None,
    ) -> RuntimeConfig:
        base = runtime or RuntimeConfig()
        updates: dict[str, object] = {}
        if max_concurrent_tasks is not None:
            updates["max_concurrent_tasks"] = max(1, max_concurrent_tasks)
        if stage_concurrency is not None:
            updates["stage_concurrency"] = {
                stage: max(1, limit) for stage, limit in stage_concurrency.items()
            }
        if provider_concurrency is not None:
            updates["provider_concurrency"] = {
                provider: max(1, limit)
                for provider, limit in provider_concurrency.items()
            }
        if provider_rate_limits is not None:
            updates["provider_rate_limits"] = {
                provider: max(1, limit)
                for provider, limit in provider_rate_limits.items()
            }
        updates["provider_token_limits"] = {
            provider: max(1, limit)
            for provider, limit in base.provider_token_limits.items()
        }
        updates["generation_retry_attempts"] = max(1, base.generation_retry_attempts)
        updates["generation_retry_delay"] = max(0.0, base.generation_retry_delay)
        updates["generation_retry_backoff"] = max(1.0, base.generation_retry_backoff)
        updates["judge_retry_attempts"] = max(1, base.judge_retry_attempts)
        updates["judge_retry_delay"] = max(0.0, base.judge_retry_delay)
        updates["judge_retry_backoff"] = max(1.0, base.judge_retry_backoff)
        if store_retry_delay is not None:
            updates["store_retry_delay"] = max(0.0, store_retry_delay)
        if store_retry_attempts is not None:
            updates["store_retry_attempts"] = max(1, store_retry_attempts)
        return base.model_copy(update=updates)
