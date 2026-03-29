"""Async execution orchestrator for Themis Phase 2."""

from __future__ import annotations

import asyncio
import json
from time import monotonic
from collections.abc import Mapping

from themis.core.config import RuntimeConfig
from themis.core.contexts import EvalScoreContext, GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.events import (
    EvaluationCompletedEvent,
    EvaluationFailedEvent,
    GenerationCompletedEvent,
    GenerationFailedEvent,
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
from themis.core.protocols import (
    CandidateReducer,
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
from themis.core.results import CaseExecutionState, CaseResult, ExecutionState, GenerationWorkItem, ProgressSnapshot, RunResult, RunStatus
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
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
        reducer: CandidateReducer | None = None,
        parser: Parser | None = None,
        metrics: list[PureMetric | LLMMetric | SelectionMetric | TraceMetric] | None = None,
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
    ) -> None:
        self.store = store
        self.generator = generator
        self.reducer = reducer
        self.parser = parser
        self.metrics = list(metrics or [])
        self.judge_models = list(judge_models or [])
        self.force_workflow_metrics = set(force_workflow_metrics or set())
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
                    self.runtime.stage_concurrency.get("generation", self.runtime.max_concurrent_tasks),
                )
            ),
            "evaluation": asyncio.Semaphore(
                max(
                    1,
                    self.runtime.stage_concurrency.get("evaluation", self.runtime.max_concurrent_tasks),
                )
            ),
        }
        self.workflow_runner = workflow_runner or DefaultWorkflowRunner(
            store=store,
            judge_models=self.judge_models,
            model_call_executor=self._execute_judge_model_call,
        )
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        self._provider_limiters: dict[str, TokenBucketRateLimiter] = {}

    async def run(self, snapshot: RunSnapshot) -> RunResult:
        stored_run = self.store.resume(snapshot.run_id)
        existing_events = stored_run.events if stored_run is not None else []
        existing_state = (
            stored_run.execution_state if stored_run is not None else ExecutionState(run_id=snapshot.run_id)
        )
        run_span = self.tracing_provider.start_span("run", {"run_id": snapshot.run_id})
        if not any(isinstance(event, RunStartedEvent) for event in existing_events):
            await self._persist_event(RunStartedEvent(run_id=snapshot.run_id))

        case_results: list[CaseResult] = []
        case_failures: list[bool] = []

        try:
            async for case_result, case_failed in self._run_cases(snapshot, existing_state):
                case_results.append(case_result)
                case_failures.append(case_failed)
            status = RunStatus.PARTIAL_FAILURE if any(case_failures) else RunStatus.COMPLETED
            await self._persist_event(RunCompletedEvent(run_id=snapshot.run_id))
            self.tracing_provider.end_span(run_span, "error" if status is RunStatus.PARTIAL_FAILURE else "ok")
            return RunResult(
                run_id=snapshot.run_id,
                status=status,
                progress=ProgressSnapshot(
                    total_cases=sum(len(dataset.cases) for dataset in snapshot.datasets),
                    completed_cases=sum(1 for failed in case_failures if not failed),
                    failed_cases=sum(1 for failed in case_failures if failed),
                ),
                cases=case_results,
            )
        except Exception as exc:
            await self._persist_event(RunFailedEvent(run_id=snapshot.run_id, error_message=str(exc)))
            self.tracing_provider.end_span(run_span, "error")
            raise

    async def _run_cases(self, snapshot: RunSnapshot, existing_state: ExecutionState):
        current_case_id: str | None = None
        current_items: list[GenerationWorkItem] = []
        case_groups: list[list[GenerationWorkItem]] = []
        async for item in self.planner.iter_work_items(snapshot):
            if current_case_id is None:
                current_case_id = item.case_id
            if item.case_id != current_case_id:
                case_groups.append(current_items)
                current_items = []
                current_case_id = item.case_id
            current_items.append(item)
        if current_items:
            case_groups.append(current_items)
        for result in await asyncio.gather(
            *(self._run_case(snapshot, items, existing_state) for items in case_groups)
        ):
            yield result

    async def _run_case(
        self,
        snapshot: RunSnapshot,
        items: list[GenerationWorkItem],
        existing_state: ExecutionState,
    ) -> tuple[CaseResult, bool]:
        case = items[0].case
        prior_case_state = existing_state.case_states.get(case.case_id, CaseExecutionState())
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
        for candidate_index, generated, failed in await asyncio.gather(*pending_generation):
            if generated is not None:
                generated_by_index[candidate_index] = generated
            had_failure = had_failure or failed

        generated_candidates = [generated_by_index[index] for index in sorted(generated_by_index)]
        if not generated_candidates and prior_case_state.reduced_candidate is None:
            return CaseResult(case_id=case.case_id), True

        reduced = prior_case_state.reduced_candidate
        if reduced is None:
            reduce_ctx = ReduceContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                candidate_ids=[candidate.candidate_id for candidate in generated_candidates],
                seed=items[0].seed,
            )
            self._notify("before_reduce", generated_candidates, reduce_ctx)
            span = self.tracing_provider.start_span("reduction", {"case_id": case.case_id})
            try:
                reduced = self._reduce_candidates(generated_candidates, reduce_ctx)
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
                return CaseResult(case_id=case.case_id, generated_candidates=generated_candidates), True

        parsed = prior_case_state.parsed_output
        if parsed is None:
            parse_ctx = ParseContext(run_id=snapshot.run_id, case_id=case.case_id, candidate_id=reduced.candidate_id)
            self._notify("before_parse", reduced, parse_ctx)
            span = self.tracing_provider.start_span("parse", {"case_id": case.case_id})
            try:
                parsed = self._parse_candidate(reduced, parse_ctx)
                self._notify("after_parse", parsed, parse_ctx)
                await self._persist_event(
                    ParseCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        result=parsed.model_dump(mode="json"),
                    )
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

        for metric, metric_kind in zip(self.metrics, snapshot.metric_kinds, strict=False):
            if (
                metric_kind != "pure"
                and metric.component_id not in self.force_workflow_metrics
                and metric.component_id in successful_scores
                and metric.component_id in workflow_executions
            ):
                continue
            if metric_kind == "pure" and metric.component_id in successful_scores:
                continue
            if metric_kind == "pure":
                score_ctx = ScoreContext(run_id=snapshot.run_id, case=case, parsed_output=parsed, seed=items[0].seed)
                self._notify("before_score", parsed, score_ctx)
                span = self.tracing_provider.start_span(
                    "score",
                    {"case_id": case.case_id, "metric_id": metric.component_id},
                )
                try:
                    score = metric.score(parsed, case, score_ctx)
                    self._notify("after_score", score, score_ctx)
                    if isinstance(score, ScoreError):
                        score_failures[metric.component_id] = score
                        successful_scores.pop(metric.component_id, None)
                        await self._persist_event(
                            ScoreFailedEvent(
                                run_id=snapshot.run_id,
                                case_id=case.case_id,
                                candidate_id=reduced.candidate_id,
                                metric_id=score.metric_id,
                                error=score.model_dump(mode="json"),
                            )
                        )
                        had_failure = True
                        self.tracing_provider.end_span(span, "error")
                        continue
                    successful_scores[metric.component_id] = score
                    score_failures.pop(metric.component_id, None)
                    await self._persist_event(
                        ScoreCompletedEvent(
                            run_id=snapshot.run_id,
                            case_id=case.case_id,
                            candidate_id=reduced.candidate_id,
                            metric_id=score.metric_id,
                            score=score.model_dump(mode="json"),
                        )
                    )
                    self.tracing_provider.end_span(span, "ok")
                except Exception as exc:
                    score_error = ScoreError(metric_id=metric.component_id, reason=str(exc))
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
                await self._persist_event(
                    EvaluationCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        metric_id=metric.component_id,
                        execution=execution.model_dump(mode="json"),
                    )
                )
                final_score = self._final_workflow_score(metric.component_id, execution)
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

        return (
            CaseResult(
                case_id=case.case_id,
                generated_candidates=generated_candidates,
                reduced_candidate=reduced,
                parsed_output=parsed,
                evaluation_executions=[
                    workflow_executions[metric.component_id]
                    for metric, metric_kind in zip(self.metrics, snapshot.metric_kinds, strict=False)
                    if metric_kind != "pure" and metric.component_id in workflow_executions
                ],
                scores=[
                    score
                    for metric, _metric_kind in zip(self.metrics, snapshot.metric_kinds, strict=False)
                    for score in [successful_scores.get(metric.component_id) or score_failures.get(metric.component_id)]
                    if score is not None
                ],
            ),
            had_failure or len(successful_scores) != len(self.metrics),
        )

    async def _generate_candidate(
        self,
        snapshot: RunSnapshot,
        case,
        item: GenerationWorkItem,
    ) -> tuple[int, GenerationResult | None, bool]:
        generate_ctx = GenerateContext(run_id=snapshot.run_id, case_id=item.case_id, seed=item.seed)
        async with self._global_semaphore:
            async with self._stage_semaphores["generation"]:
                provider_key = self._provider_key()
                provider_semaphore = self._provider_semaphore(provider_key) if provider_key is not None else None
                provider_limiter = self._provider_limiter(provider_key) if provider_key is not None else None
                if provider_semaphore is not None:
                    await provider_semaphore.acquire()
                if provider_limiter is not None:
                    await provider_limiter.acquire()
                try:
                    self._notify("before_generate", case, generate_ctx)
                    span = self.tracing_provider.start_span("generation", {"case_id": item.case_id})
                    try:
                        generated = await self.generator.generate(case, generate_ctx)
                        await self._update_rate_limit(provider_key, generated.artifacts)
                        self._notify("after_generate", generated, generate_ctx)
                        blob_ref = await self._store_blob(
                            json.dumps(generated.model_dump(mode="json"), sort_keys=True).encode("utf-8"),
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
                        self.tracing_provider.end_span(span, "ok")
                        return item.candidate_index, generated, False
                    except Exception as exc:
                        await self._persist_event(
                            GenerationFailedEvent(
                                run_id=snapshot.run_id,
                                case_id=item.case_id,
                                candidate_id=item.candidate_id,
                                error_message=str(exc),
                            )
                        )
                        self.tracing_provider.end_span(span, "error")
                        return item.candidate_index, None, True
                finally:
                    if provider_semaphore is not None:
                        provider_semaphore.release()

    def _reduce_candidates(
        self,
        generated_candidates: list[GenerationResult],
        reduce_ctx: ReduceContext,
    ) -> ReducedCandidate:
        if self.reducer is None:
            candidate = generated_candidates[0]
            return ReducedCandidate(
                candidate_id=candidate.candidate_id,
                source_candidate_ids=[candidate.candidate_id],
                final_output=candidate.final_output,
            )
        return self.reducer.reduce(generated_candidates, reduce_ctx)

    def _parse_candidate(self, reduced: ReducedCandidate, parse_ctx: ParseContext) -> ParsedOutput:
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
            raise WorkflowBuildError("Workflow-backed metrics require at least one judge model")
        return EvalScoreContext(
            run_id=snapshot.run_id,
            case=case,
            parsed_output=parsed,
            seed=seed,
            judge_model_ref=snapshot.identity.judge_model_refs[0],
            judge_seed=seed,
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
            winner_id = reduced.source_candidate_ids[0] if reduced.source_candidate_ids else generated_candidates[0].candidate_id
            winner = next(
                (candidate for candidate in generated_candidates if candidate.candidate_id == winner_id),
                generated_candidates[0],
            )
            if winner.trace is not None:
                return TraceSubject(trace=WorkflowTrace(trace_id=f"{winner.candidate_id}:trace", steps=winner.trace))
            if winner.conversation is not None:
                return ConversationSubject(
                    conversation=ConversationTrace(
                        trace_id=f"{winner.candidate_id}:conversation",
                        messages=winner.conversation,
                    )
                )
            raise WorkflowBuildError("Trace metrics require a trace or conversation on the winning candidate")
        raise WorkflowBuildError(f"Unsupported metric kind: {metric_kind}")

    def _final_workflow_score(
        self,
        metric_id: str,
        execution,
    ) -> Score:
        if execution.aggregation_output is not None and isinstance(execution.aggregation_output.value, (int, float)):
            return Score(
                metric_id=metric_id,
                value=float(execution.aggregation_output.value),
                details=execution.aggregation_output.details,
            )
        if execution.scores:
            return execution.scores[-1]
        return Score(metric_id=metric_id, value=0.0)

    async def _execute_judge_model_call(
        self,
        judge_model: JudgeModel,
        prompt: str,
        seed: int | None,
    ) -> JudgeResponse:
        async with self._global_semaphore:
            async with self._stage_semaphores["evaluation"]:
                provider_key = getattr(judge_model, "provider_key", None)
                provider_semaphore = self._provider_semaphore(provider_key) if provider_key is not None else None
                provider_limiter = self._provider_limiter(provider_key) if provider_key is not None else None
                if provider_semaphore is not None:
                    await provider_semaphore.acquire()
                if provider_limiter is not None:
                    await provider_limiter.acquire()
                try:
                    return await judge_model.judge(prompt, seed=seed)
                finally:
                    if provider_semaphore is not None:
                        provider_semaphore.release()

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
                self.runtime.provider_concurrency.get(provider_key, self.runtime.max_concurrent_tasks),
            )
            self._provider_semaphores[provider_key] = asyncio.Semaphore(limit)
        return self._provider_semaphores[provider_key]

    def _provider_limiter(self, provider_key: str) -> TokenBucketRateLimiter:
        if provider_key not in self._provider_limiters:
            requests_per_minute = self.runtime.provider_rate_limits.get(
                provider_key,
                DEFAULT_PROVIDER_RATE_LIMIT,
            )
            self._provider_limiters[provider_key] = TokenBucketRateLimiter(requests_per_minute)
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
            updates["stage_concurrency"] = {stage: max(1, limit) for stage, limit in stage_concurrency.items()}
        if provider_concurrency is not None:
            updates["provider_concurrency"] = {
                provider: max(1, limit) for provider, limit in provider_concurrency.items()
            }
        if provider_rate_limits is not None:
            updates["provider_rate_limits"] = {
                provider: max(1, limit) for provider, limit in provider_rate_limits.items()
            }
        if store_retry_delay is not None:
            updates["store_retry_delay"] = max(0.0, store_retry_delay)
        if store_retry_attempts is not None:
            updates["store_retry_attempts"] = max(1, store_retry_attempts)
        return base.model_copy(update=updates)
