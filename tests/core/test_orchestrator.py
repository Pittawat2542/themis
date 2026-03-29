from __future__ import annotations

import pytest

from themis.core.builtins import (
    resolve_generator_component,
    resolve_judge_model_component,
    resolve_metric_component,
    resolve_parser_component,
    resolve_reducer_component,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import EvalScoreContext, GenerateContext, ScoreContext
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset, GenerationResult, Message, ParsedOutput, ScoreError, TraceStep
from themis.core.orchestrator import Orchestrator
from themis.core.results import RunStatus
from themis.core.stores.memory import InMemoryRunStore
from themis.core.tracing import NoOpTracingProvider
from themis.core.workflows import EvalStep, JudgeResponse


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

    def before_judge(self, subject, ctx) -> None:
        del subject, ctx
        self.calls.append("before_judge")

    def after_judge(self, execution, ctx) -> None:
        del execution, ctx
        self.calls.append("after_judge")

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


class ErrorMetric:
    component_id = "metric/error"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-error"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> ScoreError:
        del parsed, case, ctx
        return ScoreError(metric_id=self.component_id, reason="judge unavailable", retryable=True)


class TracedGenerator:
    component_id = "generator/traced"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-traced"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        answer = case.expected_output if case.expected_output is not None else case.input
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed}",
            final_output=answer,
            trace=[
                TraceStep(
                    step_name="draft",
                    step_type="tool_call",
                    output={"seed": ctx.seed or 0},
                )
            ],
            conversation=[Message(role="assistant", content=answer)],
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
            EvalStep(step_type="parse_judgment", config={"label_scores": {"pass": 1.0, "fail": 0.0}}),
            EvalStep(step_type="emit_score"),
            EvalStep(step_type="aggregate_scores", config={"method": "mean"}),
        ]


class RecordingLLMMetric:
    component_id = "metric/llm"
    version = "1.0"
    metric_family = "llm"

    def __init__(self) -> None:
        self.subject = None

    def fingerprint(self) -> str:
        return "metric-llm"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del ctx
        self.subject = subject
        return DemoWorkflow()


class RecordingSelectionMetric:
    component_id = "metric/select"
    version = "1.0"
    metric_family = "selection"

    def __init__(self) -> None:
        self.subject = None

    def fingerprint(self) -> str:
        return "metric-select"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del ctx
        self.subject = subject
        return DemoWorkflow()


class RecordingTraceMetric:
    component_id = "metric/trace"
    version = "1.0"
    metric_family = "trace"

    def __init__(self) -> None:
        self.subject = None

    def fingerprint(self) -> str:
        return "metric-trace"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del ctx
        self.subject = subject
        return DemoWorkflow()


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


@pytest.mark.asyncio
async def test_orchestrator_marks_score_errors_as_partial_failures_and_error_spans() -> None:
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
        metrics=[ErrorMetric()],
        tracing_provider=tracer,
    )

    result = await orchestrator.run(snapshot)

    assert result.status is RunStatus.PARTIAL_FAILURE
    assert result.progress.completed_cases == 0
    assert result.progress.failed_cases == 1
    assert ("score", "error") in tracer.ended
    assert ("run", "error") in tracer.ended


@pytest.mark.asyncio
async def test_orchestrator_executes_mixed_metric_runs_and_routes_subjects() -> None:
    llm_metric = RecordingLLMMetric()
    selection_metric = RecordingSelectionMetric()
    trace_metric = RecordingTraceMetric()
    experiment = Experiment(
        generation=GenerationConfig(
            generator=TracedGenerator(),
            candidate_policy={"num_samples": 2},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo", llm_metric, selection_metric, trace_metric],
            parsers=["parser/demo"],
            judge_models=["judge/demo"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        seeds=[7, 11],
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()
    subscriber = RecordingSubscriber()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=TracedGenerator(),
        reducer=resolve_reducer_component(experiment.generation.reducer),
        parser=resolve_parser_component(experiment.evaluation.parsers[0]),
        metrics=[resolve_metric_component(metric) for metric in experiment.evaluation.metrics],
        judge_models=[resolve_judge_model_component("judge/demo")],
        subscribers=[subscriber],
    )

    result = await orchestrator.run(snapshot)
    events = store.query_events(snapshot.run_id)

    assert result.status is RunStatus.COMPLETED
    assert sorted(score.metric_id for score in result.cases[0].scores) == [
        "metric/demo",
        "metric/llm",
        "metric/select",
        "metric/trace",
    ]
    assert llm_metric.subject.candidates[0].candidate_id == "case-1-reduced"
    assert [candidate.candidate_id for candidate in selection_metric.subject.candidates] == [
        "case-1-candidate-7",
        "case-1-candidate-11",
    ]
    assert trace_metric.subject.trace.steps[0].output["seed"] == 7
    assert "before_judge" in subscriber.calls
    assert "after_judge" in subscriber.calls
    assert any(isinstance(event, EvaluationCompletedEvent) for event in events)
