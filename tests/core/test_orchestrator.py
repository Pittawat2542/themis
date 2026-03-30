from __future__ import annotations

import asyncio

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
    StepCompletedEvent,
    StepStartedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import (
    Case,
    Dataset,
    GenerationResult,
    Message,
    ParsedOutput,
    ReducedCandidate,
    Score,
    ScoreError,
    TraceStep,
)
from themis.core.orchestrator import Orchestrator
from themis.core.results import RunStatus
from themis.core.stores.memory import InMemoryRunStore
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject
from themis.core.tracing import NoOpTracingProvider
from themis.core.workflows import AggregationResult, JudgeCall, JudgeResponse, ParsedJudgment, RenderedJudgePrompt


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
    metric_family = "pure"

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


class DemoJudgeWorkflow:
    component_id = "workflow/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-demo"

    def judge_calls(self) -> list[JudgeCall]:
        return [JudgeCall(call_id="call-0", judge_model_id="builtin/demo_judge")]

    def render_prompt(self, call: JudgeCall, subject, ctx: EvalScoreContext) -> RenderedJudgePrompt:
        del call, ctx
        if hasattr(subject, "candidates"):
            content = f"Grade {subject.candidates[0].final_output}"
        elif hasattr(subject, "trace"):
            content = f"Grade {subject.trace.model_dump(mode='json')}"
        else:
            content = f"Grade {subject.conversation.model_dump(mode='json')}"
        return RenderedJudgePrompt(prompt_id="prompt-0", content=content)

    def parse_judgment(self, call: JudgeCall, response: JudgeResponse, ctx: EvalScoreContext) -> ParsedJudgment:
        del call, ctx
        label = response.raw_response.strip().split()[0].lower()
        return ParsedJudgment(label=label, score=1.0 if label == "pass" else 0.0)

    def score_judgment(self, call: JudgeCall, judgment: ParsedJudgment, ctx: EvalScoreContext) -> Score | None:
        del call, ctx
        return Score(metric_id="metric/llm", value=float(judgment.score or 0.0), details={"label": judgment.label})

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        return AggregationResult(method="mean", value=sum(score.value for score in scores) / len(scores))


class PairwiseSelectionWorkflow:
    component_id = "workflow/select"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-select"

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(
                call_id="call-a-vs-b",
                judge_model_id="builtin/demo_judge",
                dimension_id="winner",
                candidate_indices=[0, 1],
            )
        ]

    def render_prompt(self, call: JudgeCall, subject, ctx: EvalScoreContext) -> RenderedJudgePrompt:
        del ctx
        return RenderedJudgePrompt(
            prompt_id=f"prompt-{call.call_id}",
            content=f"A={subject.candidates[0].final_output}; B={subject.candidates[1].final_output}",
        )

    def parse_judgment(self, call: JudgeCall, response: JudgeResponse, ctx: EvalScoreContext) -> ParsedJudgment:
        del call, ctx
        return ParsedJudgment(label=response.raw_response.strip().lower())

    def score_judgment(self, call: JudgeCall, judgment: ParsedJudgment, ctx: EvalScoreContext) -> Score | None:
        del call, ctx
        return Score(metric_id="metric/select", value=1.0 if judgment.label == "a" else 0.0)

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del scores, ctx
        winner = max(sorted({judgment.label for judgment in judgments}), key=[judgment.label for judgment in judgments].count)
        return AggregationResult(method="majority_vote", value=winner, details={"winner": winner})


class RecordingLLMMetric:
    component_id = "metric/llm"
    version = "1.0"
    metric_family = "llm"

    def __init__(self) -> None:
        self.subject: CandidateSetSubject | None = None
        self.ctx: EvalScoreContext | None = None

    def fingerprint(self) -> str:
        return "metric-llm"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        self.subject = subject
        self.ctx = ctx
        return DemoJudgeWorkflow()


class RecordingSelectionMetric:
    component_id = "metric/select"
    version = "1.0"
    metric_family = "selection"

    def __init__(self) -> None:
        self.subject: CandidateSetSubject | None = None

    def fingerprint(self) -> str:
        return "metric-select"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del ctx
        self.subject = subject
        return PairwiseSelectionWorkflow()


class RecordingTraceMetric:
    component_id = "metric/trace"
    version = "1.0"
    metric_family = "trace"

    def __init__(self) -> None:
        self.subject: TraceSubject | ConversationSubject | None = None

    def fingerprint(self) -> str:
        return "metric-trace"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del ctx
        self.subject = subject
        return DemoJudgeWorkflow()


class FlakyJudgeModel:
    version = "1.0"

    def __init__(self, component_id: str, *, fail: bool = False) -> None:
        self.component_id = component_id
        self._fail = fail

    def fingerprint(self) -> str:
        return f"{self.component_id}-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del prompt, seed
        if self._fail:
            raise ValueError(f"{self.component_id} timeout")
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response="pass",
        )


class PartialFailureWorkflow:
    component_id = "workflow/partial"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-partial"

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(call_id="call-ok", judge_model_id="judge/ok"),
            JudgeCall(call_id="call-fail", judge_model_id="judge/fail"),
        ]

    def render_prompt(self, call: JudgeCall, subject, ctx: EvalScoreContext) -> RenderedJudgePrompt:
        del subject, ctx
        return RenderedJudgePrompt(prompt_id=f"prompt-{call.call_id}", content=call.call_id)

    def parse_judgment(self, call: JudgeCall, response: JudgeResponse, ctx: EvalScoreContext) -> ParsedJudgment:
        del call, ctx
        return ParsedJudgment(label=response.raw_response, score=1.0)

    def score_judgment(self, call: JudgeCall, judgment: ParsedJudgment, ctx: EvalScoreContext) -> Score | None:
        del call, ctx
        return Score(metric_id="metric/partial", value=float(judgment.score or 0.0))

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        if not scores:
            return None
        return AggregationResult(method="mean", value=sum(score.value for score in scores) / len(scores))


class PartialFailureMetric:
    component_id = "metric/partial"
    version = "1.0"
    metric_family = "llm"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "metric-partial"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del subject, ctx
        self.calls += 1
        return PartialFailureWorkflow()


class AwaitedReducer:
    component_id = "reducer/awaited"
    version = "1.0"

    def __init__(self) -> None:
        self.awaited = False

    def fingerprint(self) -> str:
        return "reducer-awaited"

    async def reduce(self, candidates: list[GenerationResult], ctx) -> ReducedCandidate:
        await asyncio.sleep(0)
        self.awaited = True
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-judge-selected",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[-1].final_output,
        )


def _experiment() -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
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
async def test_orchestrator_awaits_async_reducers() -> None:
    experiment = _experiment()
    snapshot = experiment.compile()
    store = InMemoryRunStore()
    reducer = AwaitedReducer()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=resolve_generator_component(experiment.generation.generator),
        reducer=reducer,
        parser=resolve_parser_component(experiment.evaluation.parsers[0]),
        metrics=[resolve_metric_component(metric) for metric in experiment.evaluation.metrics],
    )

    result = await orchestrator.run(snapshot)

    assert reducer.awaited is True
    assert result.cases[0].reduced_candidate is not None
    assert result.cases[0].reduced_candidate.candidate_id == "case-1-judge-selected"


@pytest.mark.asyncio
async def test_orchestrator_executes_mixed_metric_runs_and_routes_subjects() -> None:
    llm_metric = RecordingLLMMetric()
    selection_metric = RecordingSelectionMetric()
    trace_metric = RecordingTraceMetric()
    experiment = Experiment(
        generation=GenerationConfig(
            generator=TracedGenerator(),
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match", llm_metric, selection_metric, trace_metric],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
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
        judge_models=[
            resolve_judge_model_component("builtin/demo_judge"),
            resolve_judge_model_component("builtin/demo_judge"),
        ],
        subscribers=[subscriber],
    )

    result = await orchestrator.run(snapshot)
    events = store.query_events(snapshot.run_id)

    assert result.status is RunStatus.COMPLETED
    assert sorted(score.metric_id for score in result.cases[0].scores) == [
        "builtin/exact_match",
        "metric/llm",
        "metric/select",
        "metric/trace",
    ]
    assert llm_metric.subject is not None
    assert llm_metric.ctx is not None
    assert selection_metric.subject is not None
    assert trace_metric.subject is not None
    assert isinstance(trace_metric.subject, TraceSubject)
    assert llm_metric.subject.candidates[0].candidate_id == "case-1-reduced"
    assert [candidate.candidate_id for candidate in selection_metric.subject.candidates] == [
        "case-1-candidate-7",
        "case-1-candidate-11",
    ]
    assert trace_metric.subject.trace.steps[0].output["seed"] == 7
    assert [ref.component_id for ref in llm_metric.ctx.judge_model_refs] == ["builtin/demo_judge", "builtin/demo_judge"]
    assert "before_judge" in subscriber.calls
    assert "after_judge" in subscriber.calls
    assert any(isinstance(event, EvaluationCompletedEvent) for event in events)


@pytest.mark.asyncio
async def test_orchestrator_persists_workflow_events_through_orchestrator_event_sink() -> None:
    llm_metric = RecordingLLMMetric()
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=[llm_metric],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge"],
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
    subscriber = RecordingSubscriber()

    store.initialize()
    store.persist_snapshot(snapshot)

    orchestrator = Orchestrator(
        store=store,
        generator=resolve_generator_component(experiment.generation.generator),
        reducer=resolve_reducer_component(experiment.generation.reducer),
        parser=resolve_parser_component(experiment.evaluation.parsers[0]),
        metrics=[llm_metric],
        judge_models=[resolve_judge_model_component("builtin/demo_judge")],
        subscribers=[subscriber],
    )

    await orchestrator.run(snapshot)

    events = store.query_events(snapshot.run_id)
    assert any(isinstance(event, StepStartedEvent) for event in events)
    assert any(isinstance(event, StepCompletedEvent) for event in events)
    assert subscriber.calls.count("on_event") >= len(events)


@pytest.mark.asyncio
async def test_orchestrator_persists_partial_workflow_failures_without_dropping_successful_calls() -> None:
    metric = PartialFailureMetric()
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=[metric],
            parsers=["builtin/json_identity"],
            judge_models=[FlakyJudgeModel("judge/ok"), FlakyJudgeModel("judge/fail", fail=True)],
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

    orchestrator = Orchestrator(
        store=store,
        generator=resolve_generator_component(experiment.generation.generator),
        reducer=resolve_reducer_component(experiment.generation.reducer),
        parser=resolve_parser_component(experiment.evaluation.parsers[0]),
        metrics=[metric],
        judge_models=[FlakyJudgeModel("judge/ok"), FlakyJudgeModel("judge/fail", fail=True)],
    )

    result = await orchestrator.run(snapshot)
    events = store.query_events(snapshot.run_id)
    case = result.cases[0]

    assert result.status is RunStatus.PARTIAL_FAILURE
    assert case.evaluation_executions[0].status == "partial_failure"
    assert case.evaluation_executions[0].aggregation_output is not None
    assert case.evaluation_executions[0].aggregation_output.value == 1.0
    assert len(case.evaluation_executions[0].judge_responses) == 1
    assert len(case.evaluation_executions[0].failures) == 1
    assert case.evaluation_executions[0].failures[0].call_id == "call-fail"
    assert case.scores[0].metric_id == "metric/partial"
    assert any(isinstance(event, EvaluationCompletedEvent) for event in events)
