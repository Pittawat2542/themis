from __future__ import annotations

import asyncio

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
    ScoreFailedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset, GenerationResult, ParsedOutput, ReducedCandidate, Score, ScoreError
from themis.core.orchestrator import Orchestrator
from themis.core.stores.memory import InMemoryRunStore
from themis.core.workflows import EvalStep, JudgeResponse


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


class RateLimitedGenerator:
    component_id = "generator/rate-limited"
    version = "1.0"
    provider_key = "openai:https://api.openai.com/v1"

    def __init__(self, *, updated_rate_limit: int | None = None) -> None:
        self.timestamps: list[float] = []
        self.updated_rate_limit = updated_rate_limit
        self.calls = 0

    def fingerprint(self) -> str:
        return "generator-rate-limited"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        import themis.core.orchestrator as orchestrator_module

        self.calls += 1
        self.timestamps.append(orchestrator_module.monotonic())
        artifacts: dict[str, object] | None = None
        if self.calls == 1 and self.updated_rate_limit is not None:
            artifacts = {
                "rate_limit": {
                    "requests_per_minute": self.updated_rate_limit,
                }
            }
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed}",
            final_output=case.expected_output,
            artifacts=artifacts,
        )


class FlakyStore(InMemoryRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_persist = True

    def persist_event(self, event) -> None:
        if self.fail_next_persist:
            self.fail_next_persist = False
            raise RuntimeError("temporary store outage")
        super().persist_event(event)


class CountingLLMMetric:
    component_id = "metric/llm"
    version = "1.0"
    metric_family = "llm"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "metric-llm"

    def build_workflow(self, subject, ctx):
        del subject, ctx
        self.calls += 1
        return DemoWorkflow()


class DemoJudgeModel:
    component_id = "judge/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "judge-demo"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response=prompt,
        )


class DemoWorkflow:
    component_id = "workflow/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-demo"

    def steps(self) -> list[EvalStep]:
        return [
            EvalStep(step_type="render_prompt", config={"template": "Grade {candidate_output}"}),
            EvalStep(step_type="model_call"),
            EvalStep(step_type="parse_judgment", config={"label_scores": {"pass": 1.0}}),
            EvalStep(step_type="emit_score"),
        ]


class SlowJudgeModel:
    component_id = "judge/slow"
    version = "1.0"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def fingerprint(self) -> str:
        return "judge-slow"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del prompt, seed
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response="pass",
        )


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


@pytest.mark.asyncio
async def test_orchestrator_retries_failed_scores_on_resume() -> None:
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
    store.persist_event(
        ScoreFailedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id=metric.component_id,
            error=ScoreError(metric_id=metric.component_id, reason="temporary", retryable=True).model_dump(mode="json"),
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


@pytest.mark.asyncio
async def test_orchestrator_applies_default_provider_rate_limit() -> None:
    generator = RateLimitedGenerator()
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
    current_time = {"value": 0.0}

    def fake_monotonic() -> float:
        return current_time["value"]

    async def fake_sleep(delay: float) -> None:
        current_time["value"] += delay

    import themis.core.orchestrator as orchestrator_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(orchestrator_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(orchestrator_module.asyncio, "sleep", fake_sleep)

    try:
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
    finally:
        monkeypatch.undo()

    assert generator.timestamps == [0.0, 1.0]


@pytest.mark.asyncio
async def test_orchestrator_updates_provider_rate_limit_from_generation_artifacts() -> None:
    generator = RateLimitedGenerator(updated_rate_limit=120)
    reducer = CountingReducer()
    parser = CountingParser()
    metric = CountingMetric()
    experiment = _experiment(
        generator=generator,
        reducer=reducer,
        parser=parser,
        metric=metric,
        num_samples=3,
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()
    current_time = {"value": 0.0}

    def fake_monotonic() -> float:
        return current_time["value"]

    async def fake_sleep(delay: float) -> None:
        current_time["value"] += delay

    import themis.core.orchestrator as orchestrator_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(orchestrator_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(orchestrator_module.asyncio, "sleep", fake_sleep)

    try:
        store.initialize()
        store.persist_snapshot(snapshot)
        orchestrator = Orchestrator(
            store=store,
            generator=generator,
            reducer=reducer,
            parser=parser,
            metrics=[metric],
            max_concurrent_tasks=3,
            stage_concurrency={"generation": 3},
        )

        await orchestrator.run(snapshot)
    finally:
        monkeypatch.undo()

    assert generator.timestamps == [0.0, 0.5, 1.0]


@pytest.mark.asyncio
async def test_orchestrator_resumes_without_reevaluating_completed_workflow_metrics() -> None:
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    metric = CountingLLMMetric()
    experiment = Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": 1},
            reducer=reducer,
        ),
        evaluation=EvaluationConfig(
            metrics=[metric],
            parsers=[parser],
            judge_models=[DemoJudgeModel()],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        seeds=[7],
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
    store.persist_event(
        EvaluationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            metric_id=metric.component_id,
            execution={
                "execution_id": "execution-1",
                "subject_kind": "candidate_set",
                "scores": [{"metric_id": metric.component_id, "value": 1.0}],
                "trace": {"trace_id": "trace-1", "steps": []},
            },
        )
    )
    store.persist_event(
        ScoreCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id=metric.component_id,
            score={"metric_id": metric.component_id, "value": 1.0},
        )
    )

    orchestrator = Orchestrator(
        store=store,
        generator=generator,
        reducer=reducer,
        parser=parser,
        metrics=[metric],
        judge_models=[DemoJudgeModel()],
    )

    await orchestrator.run(snapshot)

    assert generator.calls == 0
    assert reducer.calls == 0
    assert parser.calls == 0
    assert metric.calls == 0


@pytest.mark.asyncio
async def test_orchestrator_respects_evaluation_concurrency_cap() -> None:
    judge_model = SlowJudgeModel()
    metric = CountingLLMMetric()
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    experiment = Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": 1},
            reducer=reducer,
        ),
        evaluation=EvaluationConfig(
            metrics=[metric],
            parsers=[parser],
            judge_models=[judge_model],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}),
                    Case(case_id="case-2", input={"question": "3+3"}, expected_output={"answer": "6"}),
                ],
            )
        ],
        seeds=[7],
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
        judge_models=[judge_model],
        max_concurrent_tasks=4,
        stage_concurrency={"evaluation": 1},
    )

    await orchestrator.run(snapshot)

    assert judge_model.max_active <= 1
