"""Async execution orchestrator for Themis Phase 2."""

from __future__ import annotations

import asyncio
import json

from themis.core.contexts import GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.events import (
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
from themis.core.models import ParsedOutput, ReducedCandidate, Score, ScoreError
from themis.core.planner import Planner
from themis.core.results import CaseExecutionState, CaseResult, ExecutionState, ProgressSnapshot, RunResult, RunStatus
from themis.core.snapshot import RunSnapshot
from themis.core.tracing import NoOpTracingProvider


class Orchestrator:
    def __init__(
        self,
        *,
        store,
        generator,
        reducer=None,
        parser=None,
        metrics=None,
        planner: Planner | None = None,
        subscribers=None,
        tracing_provider=None,
        max_concurrent_tasks: int = 1,
        stage_concurrency: dict[str, int] | None = None,
        provider_limits: dict[str, int] | None = None,
        store_retry_delay: float = 0.01,
        store_retry_attempts: int = 5,
    ) -> None:
        self.store = store
        self.generator = generator
        self.reducer = reducer
        self.parser = parser
        self.metrics = list(metrics or [])
        self.planner = planner or Planner()
        self.subscribers = list(subscribers or [])
        self.tracing_provider = tracing_provider or NoOpTracingProvider()
        self.max_concurrent_tasks = max(1, max_concurrent_tasks)
        self.stage_concurrency = stage_concurrency or {}
        self.provider_limits = provider_limits or {}
        self.store_retry_delay = store_retry_delay
        self.store_retry_attempts = store_retry_attempts
        self._global_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self._stage_semaphores = {
            "generation": asyncio.Semaphore(
                max(1, self.stage_concurrency.get("generation", self.max_concurrent_tasks))
            )
        }
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}

    async def run(self, snapshot: RunSnapshot) -> RunResult:
        stored_run = self.store.resume(snapshot.run_id)
        existing_events = stored_run.events if stored_run is not None else []
        existing_state = stored_run.execution_state if stored_run is not None else ExecutionState(run_id=snapshot.run_id)
        run_span = self.tracing_provider.start_span("run", {"run_id": snapshot.run_id})
        if not any(isinstance(event, RunStartedEvent) for event in existing_events):
            await self._persist_event(RunStartedEvent(run_id=snapshot.run_id))
        case_results: list[CaseResult] = []
        had_failures = False

        try:
            async for case_result, case_failed in self._run_cases(snapshot, existing_state):
                case_results.append(case_result)
                had_failures = had_failures or case_failed
            status = RunStatus.PARTIAL_FAILURE if had_failures else RunStatus.COMPLETED
            await self._persist_event(RunCompletedEvent(run_id=snapshot.run_id))
            self.tracing_provider.end_span(run_span, "ok")
            return RunResult(
                run_id=snapshot.run_id,
                status=status,
                progress=ProgressSnapshot(
                    total_cases=sum(len(dataset.cases) for dataset in snapshot.datasets),
                    completed_cases=sum(1 for case in case_results if case.parsed_output is not None),
                    failed_cases=sum(1 for case in case_results if case.parsed_output is None),
                ),
                cases=case_results,
            )
        except Exception as exc:
            await self._persist_event(RunFailedEvent(run_id=snapshot.run_id, error_message=str(exc)))
            self.tracing_provider.end_span(run_span, "error")
            raise

    async def _run_cases(self, snapshot: RunSnapshot, existing_state: ExecutionState):
        current_case_id: str | None = None
        current_items = []
        async for item in self.planner.iter_work_items(snapshot):
            if current_case_id is None:
                current_case_id = item.case_id
            if item.case_id != current_case_id:
                yield await self._run_case(snapshot, current_items, existing_state)
                current_items = []
                current_case_id = item.case_id
            current_items.append(item)
        if current_items:
            yield await self._run_case(snapshot, current_items, existing_state)

    async def _run_case(
        self,
        snapshot: RunSnapshot,
        items: list,
        existing_state: ExecutionState,
    ) -> tuple[CaseResult, bool]:
        case = items[0].case
        prior_case_state = existing_state.case_states.get(case.case_id, CaseExecutionState())
        generated_by_index = dict(prior_case_state.generated_candidates_by_index)
        had_failure = False

        pending_generation = []
        for item in items:
            if item.candidate_index not in generated_by_index:
                pending_generation.append(self._generate_candidate(snapshot, case, item))

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
            try:
                self._notify("before_reduce", generated_candidates, reduce_ctx)
                span = self.tracing_provider.start_span("reduction", {"case_id": case.case_id})
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
                return CaseResult(case_id=case.case_id, generated_candidates=generated_candidates), True

        parsed = prior_case_state.parsed_output
        if parsed is None:
            parse_ctx = ParseContext(run_id=snapshot.run_id, case_id=case.case_id, candidate_id=reduced.candidate_id)
            try:
                self._notify("before_parse", reduced, parse_ctx)
                span = self.tracing_provider.start_span("parse", {"case_id": case.case_id})
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
                return CaseResult(
                    case_id=case.case_id,
                    generated_candidates=generated_candidates,
                    reduced_candidate=reduced,
                ), True

        scores_by_metric = dict(prior_case_state.scores)
        for metric in self.metrics:
            if metric.component_id in scores_by_metric:
                continue
            score_ctx = ScoreContext(run_id=snapshot.run_id, case=case, parsed_output=parsed, seed=items[0].seed)
            self._notify("before_score", parsed, score_ctx)
            span = self.tracing_provider.start_span("score", {"case_id": case.case_id, "metric_id": metric.component_id})
            score = metric.score(parsed, case, score_ctx)
            self._notify("after_score", score, score_ctx)
            if isinstance(score, ScoreError):
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
            else:
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
            scores_by_metric[metric.component_id] = score

        return (
            CaseResult(
                case_id=case.case_id,
                generated_candidates=generated_candidates,
                reduced_candidate=reduced,
                parsed_output=parsed,
                scores=list(scores_by_metric.values()),
            ),
            had_failure,
        )

    async def _generate_candidate(self, snapshot: RunSnapshot, case, item):
        generate_ctx = GenerateContext(run_id=snapshot.run_id, case_id=item.case_id, seed=item.seed)
        async with self._global_semaphore:
            async with self._stage_semaphores["generation"]:
                provider_key = getattr(self.generator, "provider_key", None)
                semaphore = self._provider_semaphore(provider_key) if provider_key is not None else None
                if semaphore is not None:
                    await semaphore.acquire()
                try:
                    self._notify("before_generate", case, generate_ctx)
                    span = self.tracing_provider.start_span("generation", {"case_id": item.case_id})
                    try:
                        generated = await self.generator.generate(case, generate_ctx)
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
                    if semaphore is not None:
                        semaphore.release()

    def _reduce_candidates(self, generated_candidates: list, reduce_ctx: ReduceContext) -> ReducedCandidate:
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

    async def _persist_event(self, event) -> None:
        for attempt in range(self.store_retry_attempts):
            try:
                self.store.persist_event(event)
                break
            except Exception:
                if attempt + 1 == self.store_retry_attempts:
                    raise
                await asyncio.sleep(self.store_retry_delay)
        self._notify("on_event", event)

    async def _store_blob(self, blob: bytes, media_type: str) -> str:
        for attempt in range(self.store_retry_attempts):
            try:
                return self.store.store_blob(blob, media_type)
            except Exception:
                if attempt + 1 == self.store_retry_attempts:
                    raise
                await asyncio.sleep(self.store_retry_delay)
        raise RuntimeError("unreachable")

    def _notify(self, method_name: str, *args) -> None:
        for subscriber in self.subscribers:
            method = getattr(subscriber, method_name, None)
            if method is not None:
                method(*args)

    def _provider_semaphore(self, provider_key: str) -> asyncio.Semaphore:
        if provider_key not in self._provider_semaphores:
            limit = max(1, self.provider_limits.get(provider_key, self.max_concurrent_tasks))
            self._provider_semaphores[provider_key] = asyncio.Semaphore(limit)
        return self._provider_semaphores[provider_key]
