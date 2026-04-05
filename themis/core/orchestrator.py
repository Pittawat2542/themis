"""Async execution orchestrator for the Themis v4 runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from time import monotonic
from collections.abc import Mapping
from typing import Literal, TypeGuard, cast

from themis.core.base import JSONValue
from themis.core.config import RuntimeConfig
from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
    SelectContext,
)
from themis.core.events import (
    EvaluationCompletedEvent,
    EvaluationFailedEvent,
    GenerationCompletedEvent,
    GenerationFailedEvent,
    SelectionCompletedEvent,
    SelectionFailedEvent,
    ParseCompletedEvent,
    ParseFailedEvent,
    ReductionCompletedEvent,
    ReductionFailedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
    ScoreFailedEvent,
)
from themis.core.models import (
    ConversationTrace,
    GenerationResult,
    ParsedOutput,
    ReducedCandidate,
    Score,
    ScoreError,
    WorkflowTrace,
)
from themis.core.planner import Planner
from themis.core.projections import build_run_result
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
    RunResult,
    RunStatus,
)
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.stores.memory import InMemoryRunStore
from themis.core.subjects import (
    ConversationSubject,
    TraceSubject,
    candidate_set_subject_for_llm_metric,
    candidate_set_subject_for_selection_metric,
)
from themis.core.tracing import NoOpTracingProvider
from themis.core.workflow_runner import DefaultWorkflowRunner, WorkflowBuildError
from themis.core.workflows import JudgeResponse

DEFAULT_PROVIDER_RATE_LIMIT = 60
WorkflowMetric = LLMMetric | SelectionMetric | TraceMetric
RuntimeMetric = PureMetric | WorkflowMetric
ConnectionLikeErrors = (ConnectionError, OSError)


def _is_pure_metric(metric: RuntimeMetric) -> TypeGuard[PureMetric]:
    return isinstance(metric, PureMetric)


def _is_workflow_metric(metric: RuntimeMetric) -> TypeGuard[WorkflowMetric]:
    return (
        isinstance(metric, LLMMetric)
        or isinstance(metric, SelectionMetric)
        or isinstance(metric, TraceMetric)
    )


class TokenBucketRateLimiter:
    """Token-bucket limiter with a single-token bucket to smooth request rate."""

    def __init__(self, requests_per_minute: int) -> None:
        self._requests_per_minute = max(1, requests_per_minute)
        self._tokens = 1.0
        self._updated_at = monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = monotonic()
                self._refill(now)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_time = (1.0 - self._tokens) / self._tokens_per_second
            await asyncio.sleep(wait_time)

    async def update_limit(self, requests_per_minute: int) -> None:
        async with self._lock:
            now = monotonic()
            self._refill(now)
            self._requests_per_minute = max(1, requests_per_minute)
            self._tokens = min(self._tokens, 1.0)

    @property
    def _tokens_per_second(self) -> float:
        return self._requests_per_minute / 60.0

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(1.0, self._tokens + elapsed * self._tokens_per_second)
        self._updated_at = now


class Orchestrator:
    def __init__(
        self,
        *,
        store: RunStore,
        generator: Generator,
        selector: CandidateSelector | None = None,
        reducer: CandidateReducer | None = None,
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
        until_stage: Literal["generate", "reduce", "parse", "score", "judge"] = "judge",
    ) -> None:
        self.store = store
        self.generator = generator
        self.selector = selector
        self.reducer = reducer
        self.parser = parser
        self.metrics = list(metrics or [])
        self.judge_models = list(judge_models or [])
        self.force_workflow_metrics = set(force_workflow_metrics or set())
        self.replay_stage = replay_stage
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
        self._global_semaphore = asyncio.Semaphore(self.runtime.max_concurrent_tasks)
        self._stage_semaphores = {
            "generation": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get(
                        "generation", self.runtime.max_concurrent_tasks
                    ),
                )
            ),
            "evaluation": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get(
                        "evaluation", self.runtime.max_concurrent_tasks
                    ),
                )
            ),
            "selection": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get(
                        "selection", self.runtime.max_concurrent_tasks
                    ),
                )
            ),
            "reduction": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get(
                        "reduction", self.runtime.max_concurrent_tasks
                    ),
                )
            ),
            "parsing": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get(
                        "parsing", self.runtime.max_concurrent_tasks
                    ),
                )
            ),
            "scoring": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get(
                        "scoring", self.runtime.max_concurrent_tasks
                    ),
                )
            ),
        }
        self.workflow_runner = workflow_runner or DefaultWorkflowRunner(
            store=store,
            judge_models=self.judge_models,
            model_call_executor=self._execute_judge_model_call,
            persist_event=self._persist_event,
        )
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        self._provider_limiters: dict[str, TokenBucketRateLimiter] = {}

    async def run(self, snapshot: RunSnapshot) -> RunResult:
        stored_run = self.store.resume(snapshot.run_id)
        existing_events = stored_run.events if stored_run is not None else []
        existing_state = (
            stored_run.execution_state
            if stored_run is not None
            else ExecutionState(run_id=snapshot.run_id)
        )
        run_span = self.tracing_provider.start_span("run", {"run_id": snapshot.run_id})
        if not any(isinstance(event, RunStartedEvent) for event in existing_events):
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
            stored_run = self.store.resume(snapshot.run_id)
            if stored_run is None:
                raise RuntimeError(f"Run disappeared from store: {snapshot.run_id}")
            return build_run_result(stored_run.snapshot, stored_run.events)
        except Exception as exc:
            await self._persist_event(
                RunFailedEvent(run_id=snapshot.run_id, error_message=str(exc))
            )
            self.tracing_provider.end_span(run_span, "error")
            raise

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
        current_case_id: str | None = None
        current_items: list[GenerationWorkItem] = []
        async for item in self.planner.iter_work_items(snapshot):
            if current_case_id is None:
                current_case_id = item.case_id
            if item.case_id != current_case_id:
                yield current_items
                current_items = []
                current_case_id = item.case_id
            current_items.append(item)
        if current_items:
            yield current_items

    async def _run_case(
        self,
        snapshot: RunSnapshot,
        items: list[GenerationWorkItem],
        existing_state: ExecutionState,
    ) -> tuple[CaseResult, bool]:
        case = items[0].case
        prior_case_state = self._replay_case_state(
            existing_state.case_states.get(case.case_id, CaseExecutionState())
        )
        generated_by_index = dict(prior_case_state.generated_candidates_by_index)
        workflow_executions = dict(prior_case_state.evaluation_executions)
        evaluation_failures = dict(prior_case_state.evaluation_failures)
        successful_scores = dict(prior_case_state.successful_scores)
        score_failures = dict(prior_case_state.score_failures)
        had_failure = False

        pending_generation = [
            self._generate_candidate(snapshot, case, item)
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
            return CaseResult(case_id=case.case_id), True
        if self.until_stage == "generate":
            return (
                CaseResult(
                    case_id=case.case_id,
                    generated_candidates=generated_candidates,
                ),
                had_failure,
            )

        selected_candidates = self._selected_candidates_from_state(
            prior_case_state, generated_candidates
        )
        if (
            self.selector is not None
            and prior_case_state.selected_candidate_ids is None
        ):
            select_ctx = SelectContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                candidate_ids=[
                    candidate.candidate_id for candidate in generated_candidates
                ],
                seed=items[0].seed,
                judge_models=list(self.judge_models),
            )
            span = self.tracing_provider.start_span(
                "selection", {"case_id": case.case_id}
            )
            try:
                selected_candidates = await self._select_candidates(
                    generated_candidates, select_ctx
                )
                if not selected_candidates:
                    raise ValueError("Candidate selector returned no candidates")
                await self._persist_event(
                    SelectionCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_ids=[
                            candidate.candidate_id for candidate in selected_candidates
                        ],
                        metadata={"selector_id": self.selector.component_id},
                    )
                )
                self.tracing_provider.end_span(span, "ok")
            except Exception as exc:
                await self._persist_event(
                    SelectionFailedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        error_message=str(exc),
                    )
                )
                self.tracing_provider.end_span(span, "error")
                return CaseResult(
                    case_id=case.case_id, generated_candidates=generated_candidates
                ), True

        reduced = prior_case_state.reduced_candidate
        if reduced is None:
            cached_reduction = self._load_stage_cache(
                "reduce",
                self._reduction_cache_key(snapshot, selected_candidates),
            )
            if isinstance(cached_reduction, dict) and isinstance(
                cached_reduction.get("result"), dict
            ):
                reduced = ReducedCandidate.model_validate(cached_reduction["result"])
                await self._persist_event(
                    ReductionCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
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
                candidate_ids=[
                    candidate.candidate_id for candidate in selected_candidates
                ],
                seed=items[0].seed,
                metadata={"selector_id": self.selector.component_id}
                if self.selector is not None
                else {},
            )
            self._notify("before_reduce", selected_candidates, reduce_ctx)
            span = self.tracing_provider.start_span(
                "reduction", {"case_id": case.case_id}
            )
            try:
                reduced = await self._reduce_candidates(selected_candidates, reduce_ctx)
                self._notify("after_reduce", reduced, reduce_ctx)
                await self._persist_event(
                    ReductionCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        source_candidate_ids=reduced.source_candidate_ids,
                        result=reduced.model_dump(mode="json"),
                    )
                )
                self._store_stage_cache(
                    "reduce",
                    self._reduction_cache_key(snapshot, selected_candidates),
                    {
                        "source_run_id": snapshot.run_id,
                        "result": reduced.model_dump(mode="json"),
                    },
                )
                self.tracing_provider.end_span(span, "ok")
            except Exception as exc:
                await self._persist_event(
                    ReductionFailedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        error_message=str(exc),
                    )
                )
                self.tracing_provider.end_span(span, "error")
                return CaseResult(
                    case_id=case.case_id, generated_candidates=generated_candidates
                ), True
        if self.until_stage == "reduce":
            return (
                CaseResult(
                    case_id=case.case_id,
                    generated_candidates=generated_candidates,
                    reduced_candidate=reduced,
                ),
                had_failure,
            )

        parsed = prior_case_state.parsed_output
        if parsed is None:
            cached_parse = self._load_stage_cache(
                "parse",
                self._parse_cache_key(snapshot, reduced),
            )
            if isinstance(cached_parse, dict) and isinstance(
                cached_parse.get("result"), dict
            ):
                parsed = ParsedOutput.model_validate(cached_parse["result"])
                await self._persist_event(
                    ParseCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        result=parsed.model_dump(mode="json"),
                        cache_hit=True,
                        source_run_id=cast(
                            str | None, cached_parse.get("source_run_id")
                        ),
                    )
                )
        if parsed is None:
            parse_ctx = ParseContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                candidate_id=reduced.candidate_id,
            )
            self._notify("before_parse", reduced, parse_ctx)
            span = self.tracing_provider.start_span("parse", {"case_id": case.case_id})
            try:
                async with self._global_semaphore:
                    async with self._stage_semaphores["parsing"]:
                        parsed = await asyncio.to_thread(
                            self._parse_candidate, reduced, parse_ctx
                        )
                self._notify("after_parse", parsed, parse_ctx)
                await self._persist_event(
                    ParseCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        result=parsed.model_dump(mode="json"),
                    )
                )
                self._store_stage_cache(
                    "parse",
                    self._parse_cache_key(snapshot, reduced),
                    {
                        "source_run_id": snapshot.run_id,
                        "result": parsed.model_dump(mode="json"),
                    },
                )
                self.tracing_provider.end_span(span, "ok")
            except Exception as exc:
                await self._persist_event(
                    ParseFailedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        error_message=str(exc),
                    )
                )
                self.tracing_provider.end_span(span, "error")
                return (
                    CaseResult(
                        case_id=case.case_id,
                        generated_candidates=generated_candidates,
                        reduced_candidate=reduced,
                    ),
                    True,
                )
        if self.until_stage == "parse":
            return (
                CaseResult(
                    case_id=case.case_id,
                    generated_candidates=generated_candidates,
                    reduced_candidate=reduced,
                    parsed_output=parsed,
                ),
                had_failure,
            )

        for metric, metric_kind in zip(
            self.metrics, snapshot.metric_kinds, strict=False
        ):
            if self.until_stage == "score" and metric_kind != "pure":
                continue
            if (
                metric_kind != "pure"
                and metric.component_id not in self.force_workflow_metrics
                and metric.component_id in successful_scores
                and metric.component_id in workflow_executions
                and workflow_executions[metric.component_id].status == "completed"
                and not workflow_executions[metric.component_id].failures
            ):
                continue
            if metric_kind == "pure" and metric.component_id in successful_scores:
                continue
            if metric_kind == "pure":
                if not _is_pure_metric(metric):
                    raise TypeError(
                        f"Metric {metric.component_id} does not implement PureMetric"
                    )
                cache_key = self._score_cache_key(snapshot, case, parsed, metric)
                cached_score = self._load_stage_cache("score", cache_key)
                if isinstance(cached_score, dict) and isinstance(
                    cached_score.get("score"), dict
                ):
                    cached_score_result = Score.model_validate(cached_score["score"])
                    successful_scores[metric.component_id] = cached_score_result
                    score_failures.pop(metric.component_id, None)
                    await self._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            case_id=case.case_id,
                            candidate_id=reduced.candidate_id,
                            metric_id=cached_score_result.metric_id,
                            score=cached_score_result.model_dump(mode="json"),
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
                    parsed_output=parsed,
                    seed=items[0].seed,
                )
                self._notify("before_score", parsed, score_ctx)
                span = self.tracing_provider.start_span(
                    "score",
                    {"case_id": case.case_id, "metric_id": metric.component_id},
                )
                try:
                    async with self._stage_semaphores["scoring"]:
                        score_result = await asyncio.to_thread(
                            _score_pure_metric,
                            metric,
                            parsed,
                            case,
                            score_ctx,
                        )
                    self._notify("after_score", score_result, score_ctx)
                    if isinstance(score_result, ScoreError):
                        score_failures[metric.component_id] = score_result
                        successful_scores.pop(metric.component_id, None)
                        await self._persist_event(
                            ScoreFailedEvent(
                                run_id=snapshot.run_id,
                                case_id=case.case_id,
                                candidate_id=reduced.candidate_id,
                                metric_id=score_result.metric_id,
                                error=score_result.model_dump(mode="json"),
                            )
                        )
                        had_failure = True
                        self.tracing_provider.end_span(span, "error")
                        continue
                    successful_scores[metric.component_id] = score_result
                    score_failures.pop(metric.component_id, None)
                    await self._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            case_id=case.case_id,
                            candidate_id=reduced.candidate_id,
                            metric_id=score_result.metric_id,
                            score=score_result.model_dump(mode="json"),
                        )
                    )
                    self._store_stage_cache(
                        "score",
                        cache_key,
                        {
                            "source_run_id": snapshot.run_id,
                            "score": score_result.model_dump(mode="json"),
                        },
                    )
                    self.tracing_provider.end_span(span, "ok")
                except Exception as exc:
                    score_error = ScoreError(
                        metric_id=metric.component_id, reason=str(exc)
                    )
                    score_failures[metric.component_id] = score_error
                    successful_scores.pop(metric.component_id, None)
                    await self._persist_event(
                        ScoreFailedEvent(
                            run_id=snapshot.run_id,
                            case_id=case.case_id,
                            candidate_id=reduced.candidate_id,
                            metric_id=metric.component_id,
                            error=score_error.model_dump(mode="json"),
                        )
                    )
                    self.tracing_provider.end_span(span, "error")
                    had_failure = True
                continue

            if not _is_workflow_metric(metric):
                raise TypeError(
                    f"Metric {metric.component_id} does not implement a workflow metric protocol"
                )
            eval_ctx = self._evaluation_context(snapshot, case, parsed, items[0].seed)
            subject = self._evaluation_subject(
                metric_kind=metric_kind,
                generated_candidates=generated_candidates,
                reduced=reduced,
            )
            self._notify("before_judge", subject, eval_ctx)
            span = self.tracing_provider.start_span(
                "judge",
                {"case_id": case.case_id, "metric_id": metric.component_id},
            )
            try:
                workflow = metric.build_workflow(subject, eval_ctx)
                execution = await self.workflow_runner.run_evaluation(
                    workflow=workflow,
                    subject=subject,
                    metric_id=metric.component_id,
                    ctx=eval_ctx,
                )
                self._notify("after_judge", execution, eval_ctx)
                workflow_executions[metric.component_id] = execution
                evaluation_failures.pop(metric.component_id, None)
                execution_blob_ref = await self._store_blob(
                    json.dumps(
                        execution.model_dump(mode="json"), sort_keys=True
                    ).encode("utf-8"),
                    "application/json",
                )
                await self._persist_event(
                    EvaluationCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        execution=execution.model_dump(mode="json"),
                        execution_blob_ref=execution_blob_ref,
                    )
                )
                final_score = self._final_workflow_score(metric.component_id, execution)
                had_failure = (
                    had_failure
                    or execution.status == "partial_failure"
                    or bool(execution.failures)
                )
                if final_score is not None:
                    successful_scores[metric.component_id] = final_score
                    score_failures.pop(metric.component_id, None)
                    await self._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            case_id=case.case_id,
                            candidate_id=reduced.candidate_id,
                            metric_id=final_score.metric_id,
                            score=final_score.model_dump(mode="json"),
                        )
                    )
                else:
                    score_error = ScoreError(
                        metric_id=metric.component_id,
                        reason="workflow execution completed without a usable final score",
                    )
                    successful_scores.pop(metric.component_id, None)
                    score_failures[metric.component_id] = score_error
                    await self._persist_event(
                        ScoreFailedEvent(
                            run_id=snapshot.run_id,
                            case_id=case.case_id,
                            candidate_id=reduced.candidate_id,
                            metric_id=metric.component_id,
                            error=score_error.model_dump(mode="json"),
                        )
                    )
                    had_failure = True
                self.tracing_provider.end_span(span, "ok")
            except (WorkflowBuildError, Exception) as exc:
                workflow_executions.pop(metric.component_id, None)
                evaluation_failures[metric.component_id] = str(exc)
                score_error = ScoreError(metric_id=metric.component_id, reason=str(exc))
                successful_scores.pop(metric.component_id, None)
                score_failures[metric.component_id] = score_error
                await self._persist_event(
                    EvaluationFailedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        error_message=str(exc),
                    )
                )
                await self._persist_event(
                    ScoreFailedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        error=score_error.model_dump(mode="json"),
                    )
                )
                self.tracing_provider.end_span(span, "error")
                had_failure = True

        expected_metric_ids = {
            metric.component_id
            for metric, metric_kind in zip(self.metrics, snapshot.metric_kinds, strict=False)
            if self.until_stage == "judge" or metric_kind == "pure"
        }

        return (
            CaseResult(
                case_id=case.case_id,
                generated_candidates=generated_candidates,
                reduced_candidate=reduced,
                parsed_output=parsed,
                evaluation_executions=[
                    workflow_executions[metric.component_id]
                    for metric, metric_kind in zip(
                        self.metrics, snapshot.metric_kinds, strict=False
                    )
                    if metric_kind != "pure"
                    and metric.component_id in workflow_executions
                ],
                scores=[
                    score
                    for metric, _metric_kind in zip(
                        self.metrics, snapshot.metric_kinds, strict=False
                    )
                    for score in [
                        successful_scores.get(metric.component_id)
                        or score_failures.get(metric.component_id)
                    ]
                    if score is not None
                ],
            ),
            had_failure or len(successful_scores) != len(expected_metric_ids),
        )

    async def _generate_candidate(
        self,
        snapshot: RunSnapshot,
        case,
        item: GenerationWorkItem,
    ) -> tuple[int, GenerationResult | None, bool]:
        generate_ctx = GenerateContext(
            run_id=snapshot.run_id,
            case_id=item.case_id,
            seed=item.seed,
            prompt_spec=snapshot.identity.generation_prompt_spec,
        )
        cache_key = self._generation_cache_key(snapshot, case, item)
        cached_generation = self._load_stage_cache("generate", cache_key)
        if isinstance(cached_generation, dict) and isinstance(
            cached_generation.get("result"), dict
        ):
            generated = GenerationResult.model_validate(cached_generation["result"])
            blob_ref = await self._store_blob(
                json.dumps(generated.model_dump(mode="json"), sort_keys=True).encode(
                    "utf-8"
                ),
                "application/json",
            )
            await self._persist_event(
                GenerationCompletedEvent(
                    run_id=snapshot.run_id,
                    case_id=item.case_id,
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
                if provider_semaphore is not None:
                    await provider_semaphore.acquire()
                if provider_limiter is not None:
                    await provider_limiter.acquire()
                try:
                    self._notify("before_generate", case, generate_ctx)
                    span = self.tracing_provider.start_span(
                        "generation", {"case_id": item.case_id}
                    )
                    try:
                        generated = await self._generate_with_retries(
                            case, generate_ctx
                        )
                        await self._update_rate_limit(provider_key, generated.artifacts)
                        self._notify("after_generate", generated, generate_ctx)
                        blob_ref = await self._store_blob(
                            json.dumps(
                                generated.model_dump(mode="json"), sort_keys=True
                            ).encode("utf-8"),
                            "application/json",
                        )
                        await self._persist_event(
                            GenerationCompletedEvent(
                                run_id=snapshot.run_id,
                                case_id=item.case_id,
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
                            GenerationFailedEvent(
                                run_id=snapshot.run_id,
                                case_id=item.case_id,
                                candidate_id=item.candidate_id,
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
        generated_candidates: list[GenerationResult],
        select_ctx: SelectContext,
    ) -> list[GenerationResult]:
        if self.selector is None:
            return generated_candidates
        async with self._global_semaphore:
            async with self._stage_semaphores["selection"]:
                return await self.selector.select(generated_candidates, select_ctx)

    async def _reduce_candidates(
        self,
        generated_candidates: list[GenerationResult],
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
        self, reduced: ReducedCandidate, parse_ctx: ParseContext
    ) -> ParsedOutput:
        if self.parser is None:
            return ParsedOutput(value=reduced.final_output)
        return self.parser.parse(reduced, parse_ctx)

    def _evaluation_context(
        self,
        snapshot: RunSnapshot,
        case,
        parsed: ParsedOutput,
        seed: int | None,
    ) -> EvalScoreContext:
        if not snapshot.identity.judge_model_refs:
            raise WorkflowBuildError(
                "Workflow-backed metrics require at least one judge model"
            )
        return EvalScoreContext(
            run_id=snapshot.run_id,
            case=case,
            parsed_output=parsed,
            seed=seed,
            judge_model_refs=list(snapshot.identity.judge_model_refs),
            judge_seed=seed,
            prompt_spec=snapshot.identity.evaluation_prompt_spec,
            eval_workflow_config=snapshot.identity.workflow_overrides,
        )

    def _evaluation_subject(
        self,
        *,
        metric_kind: str,
        generated_candidates: list[GenerationResult],
        reduced: ReducedCandidate,
    ):
        if metric_kind == "llm":
            reduced_candidate = GenerationResult(
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
            raise WorkflowBuildError(
                "Trace metrics require a trace or conversation on the winning candidate"
            )
        raise WorkflowBuildError(f"Unsupported metric kind: {metric_kind}")

    def _final_workflow_score(
        self,
        metric_id: str,
        execution,
    ) -> Score | None:
        if execution.aggregation_output is not None and isinstance(
            execution.aggregation_output.value, (int, float)
        ):
            return Score(
                metric_id=metric_id,
                value=float(execution.aggregation_output.value),
                details=execution.aggregation_output.details,
            )
        if not execution.failures and len(execution.scores) == 1:
            return execution.scores[-1]
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
                if provider_semaphore is not None:
                    await provider_semaphore.acquire()
                if provider_limiter is not None:
                    await provider_limiter.acquire()
                try:
                    return await self._judge_with_retries(
                        judge_model, prompt, seed=seed
                    )
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
        successful_scores = dict(case_state.successful_scores)
        score_failures = dict(case_state.score_failures)

        if self.replay_stage == "judge":
            for metric_id in workflow_metric_ids:
                successful_scores.pop(metric_id, None)
                score_failures.pop(metric_id, None)
            return case_state.model_copy(
                update={
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "successful_scores": successful_scores,
                    "score_failures": score_failures,
                }
            )

        if self.replay_stage == "score":
            return case_state.model_copy(
                update={
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "successful_scores": {},
                    "score_failures": {},
                }
            )

        if self.replay_stage == "parse":
            return case_state.model_copy(
                update={
                    "parsed_output": None,
                    "parse_error": None,
                    "evaluation_executions": {},
                    "evaluation_execution_blob_refs": {},
                    "evaluation_failures": {},
                    "successful_scores": {},
                    "score_failures": {},
                }
            )

        return case_state.model_copy(
            update={
                "reduced_candidate": None,
                "reduction_error": None,
                "parsed_output": None,
                "parse_error": None,
                "evaluation_executions": {},
                "evaluation_execution_blob_refs": {},
                "evaluation_failures": {},
                "successful_scores": {},
                "score_failures": {},
            }
        )

    def _metric_kinds(self) -> list[str]:
        return [
            "pure" if _is_pure_metric(metric) else "workflow" for metric in self.metrics
        ]

    def _selected_candidates_from_state(
        self,
        case_state: CaseExecutionState,
        generated_candidates: list[GenerationResult],
    ) -> list[GenerationResult]:
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

    async def _generate_with_retries(
        self, case, generate_ctx: GenerateContext
    ) -> GenerationResult:
        retry_history: list[dict[str, JSONValue]] = []
        for attempt in range(self.runtime.generation_retry_attempts):
            try:
                generated = await self.generator.generate(case, generate_ctx)
                if retry_history:
                    artifacts = dict(generated.artifacts or {})
                    artifacts["retry_history"] = cast(JSONValue, retry_history)
                    generated = generated.model_copy(update={"artifacts": artifacts})
                return generated
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
        for attempt in range(self.runtime.store_retry_attempts):
            try:
                self.store.persist_event(event)
                break
            except Exception:
                if attempt + 1 == self.runtime.store_retry_attempts:
                    raise
                await asyncio.sleep(self.runtime.store_retry_delay)
        self._notify("on_event", event)

    async def _store_blob(self, blob: bytes, media_type: str) -> str:
        for attempt in range(self.runtime.store_retry_attempts):
            try:
                return self.store.store_blob(blob, media_type)
            except Exception:
                if attempt + 1 == self.runtime.store_retry_attempts:
                    raise
                await asyncio.sleep(self.runtime.store_retry_delay)
        raise RuntimeError("unreachable")

    def _notify(self, method_name: str, *args) -> None:
        for subscriber in self.subscribers:
            method = getattr(subscriber, method_name, None)
            if method is not None:
                method(*args)

    def _provider_key(self) -> str | None:
        return getattr(self.generator, "provider_key", None)

    def _provider_semaphore(self, provider_key: str) -> asyncio.Semaphore:
        if provider_key not in self._provider_semaphores:
            limit = max(
                1,
                self.runtime.provider_concurrency.get(
                    provider_key, self.runtime.max_concurrent_tasks
                ),
            )
            self._provider_semaphores[provider_key] = asyncio.Semaphore(limit)
        return self._provider_semaphores[provider_key]

    def _provider_limiter(self, provider_key: str) -> TokenBucketRateLimiter:
        if provider_key not in self._provider_limiters:
            requests_per_minute = self.runtime.provider_rate_limits.get(
                provider_key,
                DEFAULT_PROVIDER_RATE_LIMIT,
            )
            self._provider_limiters[provider_key] = TokenBucketRateLimiter(
                requests_per_minute
            )
        return self._provider_limiters[provider_key]

    async def _update_rate_limit(
        self,
        provider_key: str | None,
        artifacts: Mapping[str, object] | None,
    ) -> None:
        if provider_key is None or not artifacts:
            return
        rate_limit = artifacts.get("rate_limit")
        if not isinstance(rate_limit, dict):
            return
        requests_per_minute = rate_limit.get("requests_per_minute")
        if isinstance(requests_per_minute, int):
            await self._provider_limiter(provider_key).update_limit(requests_per_minute)

    def _load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        if isinstance(self.store, InMemoryRunStore):
            return None
        return self.store.load_stage_cache(stage_name, cache_key)

    def _store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        if isinstance(self.store, InMemoryRunStore):
            return
        self.store.store_stage_cache(stage_name, cache_key, payload)

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
        self, snapshot: RunSnapshot, candidates: list[GenerationResult]
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
                "candidate_hashes": [candidate.compute_hash() for candidate in candidates],
            }
        )

    def _parse_cache_key(self, snapshot: RunSnapshot, reduced: ReducedCandidate) -> str:
        return _stable_hash(
            {
                "stage": "parse",
                "reduced_hash": reduced.compute_hash(),
                "parser_ref": snapshot.component_refs.parsers[0].model_dump(mode="json")
                if snapshot.component_refs.parsers
                else None,
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


def _classify_retryable_error(exc: Exception) -> dict[str, JSONValue] | None:
    if bool(getattr(exc, "retryable", False)):
        return {"reason": "explicit_retryable"}
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return {"reason": "timeout"}
    if isinstance(exc, ConnectionLikeErrors):
        return {"reason": "connection"}
    status_code = getattr(exc, "status_code", None)
    retry_after_s = getattr(exc, "retry_after_s", None)
    if status_code == 429:
        payload: dict[str, JSONValue] = {"reason": "rate_limit"}
        if isinstance(retry_after_s, (int, float)):
            payload["retry_after_s"] = float(retry_after_s)
        return payload
    if isinstance(status_code, int) and 500 <= status_code < 600:
        payload = {"reason": "server_error"}
        if isinstance(retry_after_s, (int, float)):
            payload["retry_after_s"] = float(retry_after_s)
        return payload
    return None


def _retry_delay_seconds(
    *,
    base_delay: float,
    backoff: float,
    attempt: int,
    retry_after_s: JSONValue | None = None,
) -> float:
    computed_delay = max(0.0, base_delay) * (max(1.0, backoff) ** attempt)
    jitter_window = computed_delay * 0.1
    jittered_delay = computed_delay
    if jitter_window > 0:
        jittered_delay = max(
            0.0,
            computed_delay + random.Random(attempt).uniform(-jitter_window, jitter_window),
        )
    if isinstance(retry_after_s, (int, float)):
        return max(jittered_delay, float(retry_after_s))
    return jittered_delay


def _score_pure_metric(
    metric: PureMetric,
    parsed: ParsedOutput,
    case,
    score_ctx: ScoreContext,
) -> Score | ScoreError:
    return metric.score(parsed, case, score_ctx)


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
