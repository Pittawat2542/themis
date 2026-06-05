from __future__ import annotations

from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    SessionConfig,
    RuntimeConfig,
    StorageConfig,
)
from themis.core.events import SessionCompletedEvent, StreamRecordedEvent
from themis.core.experiment import Experiment
from themis.core.models import (
    Case,
    Dataset,
    GenerationResult,
    ParsedOutput,
    ReducedCandidate,
    MetricResult,
    Message,
    SessionResult,
    SessionTurn,
    StreamEvent,
)
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.sqlite import SqliteRunStore


def _experiment() -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )


def test_run_can_stop_after_generation_and_mark_completion_stage() -> None:
    experiment = _experiment()
    store = InMemoryRunStore()

    result = experiment.run(store=store, until_stage="generate")
    stored = store.resume(result.run_id)

    assert result.status.value == "completed"
    assert result.completed_through_stage == "generate"
    assert result.cases[0].generated_candidates
    assert result.cases[0].reduced_candidate is None
    assert stored is not None
    assert stored.execution_state.completed_through_stage == "generate"


def test_full_run_after_stage_limited_generation_resumes_downstream_work() -> None:
    experiment = _experiment()
    store = InMemoryRunStore()

    generated_only = experiment.run(store=store, until_stage="generate")
    completed = experiment.run(store=store)

    assert generated_only.completed_through_stage == "generate"
    assert completed.completed_through_stage == "judge"
    assert completed.cases[0].reduced_candidate is not None
    assert completed.cases[0].metric_results[0].metric_id == "builtin/exact_match"


class CountingGenerator:
    component_id = "generator/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "generator-counting"

    async def generate(self, case: Case, ctx) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed}",
            final_output=case.expected_output,
        )


class CountingReducer:
    component_id = "reducer/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "reducer-counting"

    async def reduce(self, candidates: list[GenerationResult], ctx) -> ReducedCandidate:
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

    def parse(self, candidate: ReducedCandidate, ctx) -> ParsedOutput:
        del ctx
        self.calls += 1
        return ParsedOutput(value=candidate.final_output, format="json")


class ExactMetric:
    component_id = "metric/exact"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-exact"

    def score(self, parsed: ParsedOutput, case: Case, ctx) -> MetricResult:
        del ctx
        return MetricResult(
            metric_id=self.component_id,
            value=float(parsed.value == case.expected_output),
        )


class AlternateMetric:
    component_id = "metric/alternate"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-alternate"

    def score(self, parsed: ParsedOutput, case: Case, ctx) -> MetricResult:
        del ctx
        return MetricResult(
            metric_id=self.component_id,
            value=float(parsed.value == case.expected_output),
        )


class StreamingSessionGenerator:
    component_id = "generator/session-streaming"
    version = "1.0"

    def fingerprint(self) -> str:
        return "session-streaming"

    async def run_session(self, case: Case, ctx) -> SessionResult:
        return SessionResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed}",
            final_output=case.expected_output,
            turns=[
                SessionTurn(
                    turn_index=0,
                    input_messages=[Message(role="user", content=case.input)],
                    output_messages=[
                        Message(role="assistant", content=case.expected_output)
                    ],
                )
            ],
            stream_events=[
                StreamEvent(
                    event_id="event-1",
                    source_stage="session",
                    event_type="token",
                    payload={"text": "4"},
                )
            ],
            termination_reason="completed",
        )


def test_session_config_runs_session_generator_and_persists_stream_events() -> None:
    store = InMemoryRunStore()
    session_config = SessionConfig(generator=StreamingSessionGenerator())
    experiment = Experiment(
        generation=session_config,
        session=session_config,
        evaluation=EvaluationConfig(metrics=[ExactMetric()], parsers=[]),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )

    result = experiment.run(store=store)
    stored = store.resume(result.run_id)

    assert result.cases[0].generated_candidates[0].turns[0].turn_index == 0
    assert result.cases[0].generated_candidates[0].termination_reason == "completed"
    assert stored is not None
    assert any(isinstance(event, SessionCompletedEvent) for event in stored.events)
    assert any(isinstance(event, StreamRecordedEvent) for event in stored.events)


def _cached_experiment(
    *,
    generator,
    reducer,
    parser,
    metric,
    store_path: str,
    existing_run_policy: str = "auto",
) -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": 1},
            reducer=reducer,
        ),
        evaluation=EvaluationConfig(metrics=[metric], parsers=[parser]),
        storage=StorageConfig(target="sqlite", kwargs={"path": store_path}),
        runtime=RuntimeConfig(existing_run_policy=existing_run_policy),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )


def test_persistent_store_reuses_stage_cache_across_runs_when_only_metric_changes(
    tmp_path,
) -> None:
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    store_path = str(tmp_path / "run.sqlite3")
    store = SqliteRunStore(store_path)
    first = _cached_experiment(
        generator=generator,
        reducer=reducer,
        parser=parser,
        metric=ExactMetric(),
        store_path=store_path,
    )
    second = _cached_experiment(
        generator=generator,
        reducer=reducer,
        parser=parser,
        metric=AlternateMetric(),
        store_path=store_path,
    )

    first.run(store=store)
    second.run(store=store)

    assert generator.calls == 1
    assert reducer.calls == 1
    assert parser.calls == 1


def test_existing_run_policy_error_rejects_rerunning_completed_run(tmp_path) -> None:
    generator = CountingGenerator()
    reducer = CountingReducer()
    parser = CountingParser()
    store_path = str(tmp_path / "run.sqlite3")
    store = SqliteRunStore(store_path)
    experiment = _cached_experiment(
        generator=generator,
        reducer=reducer,
        parser=parser,
        metric=ExactMetric(),
        store_path=store_path,
        existing_run_policy="error",
    )

    experiment.run(store=store)

    try:
        experiment.run(store=store)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected ValueError when completed run already exists")
