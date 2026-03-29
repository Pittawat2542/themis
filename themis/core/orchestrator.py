"""Async execution orchestrator for Themis Phase 2."""

from __future__ import annotations

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
from themis.core.results import CaseResult, ProgressSnapshot, RunResult, RunStatus
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
    ) -> None:
        self.store = store
        self.generator = generator
        self.reducer = reducer
        self.parser = parser
        self.metrics = list(metrics or [])
        self.planner = planner or Planner()
        self.subscribers = list(subscribers or [])
        self.tracing_provider = tracing_provider or NoOpTracingProvider()

    async def run(self, snapshot: RunSnapshot) -> RunResult:
        run_span = self.tracing_provider.start_span("run", {"run_id": snapshot.run_id})
        self._persist_event(RunStartedEvent(run_id=snapshot.run_id))
        case_results: list[CaseResult] = []
        had_failures = False

        try:
            async for case_result, case_failed in self._run_cases(snapshot):
                case_results.append(case_result)
                had_failures = had_failures or case_failed
            status = RunStatus.PARTIAL_FAILURE if had_failures else RunStatus.COMPLETED
            self._persist_event(RunCompletedEvent(run_id=snapshot.run_id))
            self.tracing_provider.end_span(run_span, "ok")
            return RunResult(
                run_id=snapshot.run_id,
                status=status,
                progress=ProgressSnapshot(
                    total_cases=len(snapshot.datasets[0].cases) if snapshot.datasets else 0,
                    completed_cases=sum(1 for case in case_results if case.parsed_output is not None),
                    failed_cases=sum(1 for case in case_results if case.parsed_output is None),
                ),
                cases=case_results,
            )
        except Exception as exc:
            self._persist_event(RunFailedEvent(run_id=snapshot.run_id, error_message=str(exc)))
            self.tracing_provider.end_span(run_span, "error")
            raise

    async def _run_cases(self, snapshot: RunSnapshot):
        current_case_id: str | None = None
        current_items = []
        async for item in self.planner.iter_work_items(snapshot):
            if current_case_id is None:
                current_case_id = item.case_id
            if item.case_id != current_case_id:
                yield await self._run_case(snapshot, current_items)
                current_items = []
                current_case_id = item.case_id
            current_items.append(item)
        if current_items:
            yield await self._run_case(snapshot, current_items)

    async def _run_case(self, snapshot: RunSnapshot, items: list) -> tuple[CaseResult, bool]:
        case = items[0].case
        generated_candidates = []
        had_failure = False

        for item in items:
            generate_ctx = GenerateContext(run_id=snapshot.run_id, case_id=item.case_id, seed=item.seed)
            self._notify("before_generate", case, generate_ctx)
            span = self.tracing_provider.start_span("generation", {"case_id": item.case_id})
            try:
                generated = await self.generator.generate(case, generate_ctx)
                self._notify("after_generate", generated, generate_ctx)
                blob_ref = self.store.store_blob(
                    json.dumps(generated.model_dump(mode="json"), sort_keys=True).encode("utf-8"),
                    "application/json",
                )
                self._persist_event(
                    GenerationCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=item.case_id,
                        candidate_id=generated.candidate_id,
                        candidate_index=item.candidate_index,
                        seed=item.seed,
                        result=generated.model_dump(mode="json"),
                        result_blob_ref=blob_ref,
                    )
                )
                self.tracing_provider.end_span(span, "ok")
                generated_candidates.append(generated)
            except Exception as exc:
                self._persist_event(
                    GenerationFailedEvent(
                        run_id=snapshot.run_id,
                        case_id=item.case_id,
                        candidate_id=item.candidate_id,
                        error_message=str(exc),
                    )
                )
                self.tracing_provider.end_span(span, "error")
                had_failure = True

        if not generated_candidates:
            return CaseResult(case_id=case.case_id), True

        try:
            reduce_ctx = ReduceContext(
                run_id=snapshot.run_id,
                case_id=case.case_id,
                candidate_ids=[candidate.candidate_id for candidate in generated_candidates],
                seed=items[0].seed,
            )
            self._notify("before_reduce", generated_candidates, reduce_ctx)
            span = self.tracing_provider.start_span("reduction", {"case_id": case.case_id})
            reduced = self._reduce_candidates(generated_candidates, reduce_ctx)
            self._notify("after_reduce", reduced, reduce_ctx)
            self._persist_event(
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
            self._persist_event(
                ReductionFailedEvent(run_id=snapshot.run_id, case_id=case.case_id, error_message=str(exc))
            )
            return CaseResult(case_id=case.case_id, generated_candidates=generated_candidates), True

        try:
            parse_ctx = ParseContext(run_id=snapshot.run_id, case_id=case.case_id, candidate_id=reduced.candidate_id)
            self._notify("before_parse", reduced, parse_ctx)
            span = self.tracing_provider.start_span("parse", {"case_id": case.case_id})
            parsed = self._parse_candidate(reduced, parse_ctx)
            self._notify("after_parse", parsed, parse_ctx)
            self._persist_event(
                ParseCompletedEvent(
                    run_id=snapshot.run_id,
                    case_id=case.case_id,
                    candidate_id=reduced.candidate_id,
                    result=parsed.model_dump(mode="json"),
                )
            )
            self.tracing_provider.end_span(span, "ok")
        except Exception as exc:
            self._persist_event(
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

        scores = []
        for metric in self.metrics:
            score_ctx = ScoreContext(run_id=snapshot.run_id, case=case, parsed_output=parsed, seed=items[0].seed)
            self._notify("before_score", parsed, score_ctx)
            span = self.tracing_provider.start_span("score", {"case_id": case.case_id, "metric_id": metric.component_id})
            score = metric.score(parsed, case, score_ctx)
            self._notify("after_score", score, score_ctx)
            if isinstance(score, ScoreError):
                self._persist_event(
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
                self._persist_event(
                    ScoreCompletedEvent(
                        run_id=snapshot.run_id,
                        case_id=case.case_id,
                        candidate_id=reduced.candidate_id,
                        metric_id=score.metric_id,
                        score=score.model_dump(mode="json"),
                    )
                )
            self.tracing_provider.end_span(span, "ok")
            scores.append(score)

        return (
            CaseResult(
                case_id=case.case_id,
                generated_candidates=generated_candidates,
                reduced_candidate=reduced,
                parsed_output=parsed,
                scores=scores,
            ),
            had_failure,
        )

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

    def _persist_event(self, event) -> None:
        self.store.persist_event(event)
        self._notify("on_event", event)

    def _notify(self, method_name: str, *args) -> None:
        for subscriber in self.subscribers:
            method = getattr(subscriber, method_name, None)
            if method is not None:
                method(*args)
