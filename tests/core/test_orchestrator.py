from __future__ import annotations

import pytest

from themis.core.builtins import (
    resolve_generator_component,
    resolve_metric_component,
    resolve_parser_component,
    resolve_reducer_component,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import (
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.orchestrator import Orchestrator
from themis.core.results import RunStatus
from themis.core.stores.memory import InMemoryRunStore
from themis.core.tracing import NoOpTracingProvider


class RecordingSubscriber:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def before_generate(self, case, ctx) -> None:
        del case, ctx
        self.calls.append("before_generate")

    def after_generate(self, result, ctx) -> None:
        del result, ctx
        self.calls.append("after_generate")

    def before_reduce(self, candidates, ctx) -> None:
        del candidates, ctx
        self.calls.append("before_reduce")

    def after_reduce(self, reduced, ctx) -> None:
        del reduced, ctx
        self.calls.append("after_reduce")

    def before_parse(self, candidate, ctx) -> None:
        del candidate, ctx
        self.calls.append("before_parse")

    def after_parse(self, parsed, ctx) -> None:
        del parsed, ctx
        self.calls.append("after_parse")

    def before_score(self, parsed, ctx) -> None:
        del parsed, ctx
        self.calls.append("before_score")

    def after_score(self, score, ctx) -> None:
        del score, ctx
        self.calls.append("after_score")

    def on_event(self, event) -> None:
        del event
        self.calls.append("on_event")


class RecordingTracer(NoOpTracingProvider):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[tuple[str, str]] = []

    def start_span(self, name: str, attributes: dict[str, object]) -> object:
        del attributes
        self.started.append(name)
        return name

    def end_span(self, span: object, status: str) -> None:
        self.ended.append((str(span), status))


def _experiment() -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 2},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
    )


@pytest.mark.asyncio
async def test_experiment_run_async_executes_builtin_pipeline_end_to_end() -> None:
    result = await _experiment().run_async()

    assert result.status is RunStatus.COMPLETED
    assert result.progress.total_cases == 1
    assert result.progress.completed_cases == 1
    assert len(result.cases) == 1
    assert len(result.cases[0].generated_candidates) == 2
    assert result.cases[0].reduced_candidate is not None
    assert result.cases[0].parsed_output is not None
    assert result.cases[0].scores[0].value == 1.0


@pytest.mark.asyncio
async def test_orchestrator_writes_stage_events_and_dispatches_hooks() -> None:
    experiment = _experiment()
    snapshot = experiment.compile()
    store = InMemoryRunStore()
    subscriber = RecordingSubscriber()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=resolve_generator_component(experiment.generation.generator),
        reducer=resolve_reducer_component(experiment.generation.reducer),
        parser=resolve_parser_component(experiment.evaluation.parsers[0]),
        metrics=[resolve_metric_component(metric) for metric in experiment.evaluation.metrics],
        subscribers=[subscriber],
    )

    result = await orchestrator.run(snapshot)
    events = store.query_events(snapshot.run_id)

    assert result.status is RunStatus.COMPLETED
    assert [type(event) for event in events] == [
        RunStartedEvent,
        GenerationCompletedEvent,
        GenerationCompletedEvent,
        ReductionCompletedEvent,
        ParseCompletedEvent,
        ScoreCompletedEvent,
        RunCompletedEvent,
    ]
    assert subscriber.calls == [
        "on_event",
        "before_generate",
        "after_generate",
        "on_event",
        "before_generate",
        "after_generate",
        "on_event",
        "before_reduce",
        "after_reduce",
        "on_event",
        "before_parse",
        "after_parse",
        "on_event",
        "before_score",
        "after_score",
        "on_event",
        "on_event",
    ]


@pytest.mark.asyncio
async def test_orchestrator_emits_tracing_spans_for_each_stage() -> None:
    experiment = _experiment()
    snapshot = experiment.compile()
    store = InMemoryRunStore()
    tracer = RecordingTracer()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=resolve_generator_component(experiment.generation.generator),
        reducer=resolve_reducer_component(experiment.generation.reducer),
        parser=resolve_parser_component(experiment.evaluation.parsers[0]),
        metrics=[resolve_metric_component(metric) for metric in experiment.evaluation.metrics],
        tracing_provider=tracer,
    )

    await orchestrator.run(snapshot)

    assert tracer.started == [
        "run",
        "generation",
        "generation",
        "reduction",
        "parse",
        "score",
    ]
    assert tracer.ended == [
        ("generation", "ok"),
        ("generation", "ok"),
        ("reduction", "ok"),
        ("parse", "ok"),
        ("score", "ok"),
        ("run", "ok"),
    ]
