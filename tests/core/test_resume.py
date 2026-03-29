from __future__ import annotations

import asyncio

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.events import GenerationCompletedEvent, ParseCompletedEvent, ReductionCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset, GenerationResult, ParsedOutput, ReducedCandidate, Score
from themis.core.orchestrator import Orchestrator
from themis.core.stores.memory import InMemoryRunStore


class ConcurrencyTrackingGenerator:
    component_id = "generator/tracking"
    version = "1.0"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def fingerprint(self) -> str:
        return "generator-tracking"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return GenerationResult(candidate_id=f"{case.case_id}-candidate-{ctx.seed}", final_output=case.expected_output)


class CountingGenerator:
    component_id = "generator/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "generator-counting"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        self.calls += 1
        return GenerationResult(candidate_id=f"{case.case_id}-candidate-{ctx.seed}", final_output=case.expected_output)


class CountingReducer:
    component_id = "reducer/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "reducer-counting"

    def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        self.calls += 1
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
        )


class CountingParser:
    component_id = "parser/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "parser-counting"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        self.calls += 1
        return ParsedOutput(value=candidate.final_output, format="json")


class CountingMetric:
    component_id = "metric/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "metric-counting"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        self.calls += 1
        return Score(metric_id=self.component_id, value=float(parsed.value == case.expected_output))


class FlakyStore(InMemoryRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_persist = True

    def persist_event(self, event) -> None:
        if self.fail_next_persist:
            self.fail_next_persist = False
            raise RuntimeError("temporary store outage")
        super().persist_event(event)


def _experiment(*, generator, reducer, parser, metric, num_samples=1) -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": num_samples},
            reducer=reducer,
        ),
        evaluation=EvaluationConfig(metrics=[metric], parsers=[parser]),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        seeds=[7] if num_samples == 1 else [7, 11, 13, 17][:num_samples],
    )


@pytest.mark.asyncio
async def test_orchestrator_respects_generation_concurrency_cap() -> None:
    generator = ConcurrencyTrackingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    metric = CountingMetric()
    experiment = _experiment(
        generator=generator,
        reducer=reducer,
        parser=parser,
        metric=metric,
        num_samples=4,
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=generator,
        reducer=reducer,
        parser=parser,
        metrics=[metric],
        max_concurrent_tasks=2,
        stage_concurrency={"generation": 2},
    )

    await orchestrator.run(snapshot)

    assert generator.max_active <= 2


@pytest.mark.asyncio
async def test_orchestrator_retries_transient_store_failures() -> None:
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    metric = CountingMetric()
    experiment = _experiment(generator=generator, reducer=reducer, parser=parser, metric=metric)
    snapshot = experiment.compile()
    store = FlakyStore()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=generator,
        reducer=reducer,
        parser=parser,
        metrics=[metric],
        store_retry_delay=0,
    )

    result = await orchestrator.run(snapshot)

    assert result.status.value in {"completed", "partial_failure"}
    assert store.query_events(snapshot.run_id)


@pytest.mark.asyncio
async def test_orchestrator_resumes_without_regenerating_completed_candidates() -> None:
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    metric = CountingMetric()
    experiment = _experiment(
        generator=generator,
        reducer=reducer,
        parser=parser,
        metric=metric,
        num_samples=2,
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-7",
            candidate_index=0,
            seed=7,
            result={"candidate_id": "case-1-candidate-7", "final_output": {"answer": "4"}},
        )
    )

    orchestrator = Orchestrator(
        store=store,
        generator=generator,
        reducer=reducer,
        parser=parser,
        metrics=[metric],
    )

    await orchestrator.run(snapshot)

    assert generator.calls == 1


@pytest.mark.asyncio
async def test_orchestrator_rescores_without_regeneration_or_reparse() -> None:
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    metric = CountingMetric()
    experiment = _experiment(generator=generator, reducer=reducer, parser=parser, metric=metric)
    snapshot = experiment.compile()
    store = InMemoryRunStore()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-7",
            candidate_index=0,
            seed=7,
            result={"candidate_id": "case-1-candidate-7", "final_output": {"answer": "4"}},
        )
    )
    store.persist_event(
        ReductionCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            source_candidate_ids=["case-1-candidate-7"],
            result={
                "candidate_id": "case-1-reduced",
                "source_candidate_ids": ["case-1-candidate-7"],
                "final_output": {"answer": "4"},
            },
        )
    )
    store.persist_event(
        ParseCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            result={"value": {"answer": "4"}, "format": "json"},
        )
    )

    orchestrator = Orchestrator(
        store=store,
        generator=generator,
        reducer=reducer,
        parser=parser,
        metrics=[metric],
    )

    await orchestrator.run(snapshot)

    assert generator.calls == 0
    assert reducer.calls == 0
    assert parser.calls == 0
    assert metric.calls == 1
