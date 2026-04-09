from __future__ import annotations

import gc
import warnings

import pytest
from typing import Any, cast

from themis import Experiment, RunSnapshot, __version__
from themis.core.base import JSONValue
from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    RuntimeConfig,
    StorageConfig,
)
from themis.core.contexts import GenerateContext
from themis.core.models import Case, Dataset, GenerationResult, Message, TraceStep
from themis.core.stores.memory import InMemoryRunStore
from themis.core.workflows import JudgeResponse
from tests.release import CURRENT_VERSION


class DummyJudgeModel:
    component_id = "judge/custom"
    version = "1.0"

    def fingerprint(self) -> str:
        return "judge-custom-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response=prompt,
        )


class MutableGenerator:
    component_id = "generator/mutable"
    version = "1.0"

    def __init__(self) -> None:
        self.fingerprint_value = "generator-fingerprint-v1"

    def fingerprint(self) -> str:
        return self.fingerprint_value

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        del ctx
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate", final_output=case.expected_output
        )


class TracedGenerator:
    component_id = "generator/traced"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-traced"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        dataset_id = ctx.dataset_id or "unknown"
        return GenerationResult(
            candidate_id=f"{dataset_id}:{case.case_id}:{ctx.seed}",
            final_output=case.expected_output,
            trace=[
                TraceStep(
                    step_name="generate",
                    step_type="model_call",
                    output={"dataset_id": dataset_id},
                )
            ],
            conversation=[Message(role="assistant", content=case.expected_output)],
        )


def test_experiment_compile_returns_snapshot() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
            judge_config={"panel_size": 1},
            judge_models=[DummyJudgeModel()],
        ),
        storage=StorageConfig(store="memory", parameters={"path": ":memory:"}),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
                revision="r1",
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version=CURRENT_VERSION,
        python_version="3.12.9",
        platform="macos",
    )

    snapshot = experiment.compile()

    assert isinstance(snapshot, RunSnapshot)
    assert snapshot.identity.dataset_refs[0].dataset_id == "dataset-1"
    assert snapshot.datasets[0].cases[0].case_id == "case-1"
    assert snapshot.component_refs.generator.component_id == "builtin/demo_generator"
    assert snapshot.component_refs.judge_models[0].component_id == "judge/custom"
    assert (
        snapshot.identity.judge_model_refs[0].fingerprint == "judge-custom-fingerprint"
    )


def test_experiment_defaults_release_provenance_to_package_version() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
    )

    assert experiment.compile().provenance.themis_version == __version__


def test_experiment_compile_captures_extended_provenance_fields() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
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
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        git_commit="abc123",
        dependency_versions={"demo": "1.2.3"},
        provider_metadata={"generator": {"provider_key": "demo"}},
    )

    snapshot = experiment.compile()

    assert snapshot.provenance.git_commit == "abc123"
    assert snapshot.provenance.dependency_versions == {"demo": "1.2.3"}
    assert snapshot.provenance.provider_metadata == {
        "generator": {"provider_key": "demo"}
    }


def test_rejudge_requires_explicit_store_for_memory_backed_runs() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
            judge_models=[DummyJudgeModel()],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        seeds=[7],
    )

    with pytest.raises(ValueError, match="original store instance"):
        experiment.rejudge()


def test_replay_requires_explicit_store_for_memory_backed_runs() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
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
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        seeds=[7],
    )

    with pytest.raises(ValueError, match="original store instance"):
        experiment.replay(stage="score")


def test_run_accepts_explicit_store_for_memory_backed_runs() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
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
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        seeds=[7],
    )
    store = InMemoryRunStore()

    result = experiment.run(store=store)

    assert result.status.value == "completed"
    assert store.resume(experiment.compile().run_id) is not None


def test_run_rejects_component_fingerprint_mismatch_after_compile() -> None:
    generator = MutableGenerator()
    experiment = Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": 1},
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
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
    )
    store = InMemoryRunStore()

    experiment.compile()
    generator.fingerprint_value = "generator-fingerprint-v2"

    with pytest.raises(RuntimeError, match="Component fingerprint mismatch"):
        experiment.run(store=store)


def test_run_rejects_component_fingerprint_mismatch_before_auto_reuse() -> None:
    generator = MutableGenerator()
    experiment = Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        runtime=RuntimeConfig(existing_run_policy="auto"),
        datasets=[
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
    )
    store = InMemoryRunStore()

    first = experiment.run(store=store)

    assert first.status.value == "completed"

    generator.fingerprint_value = "generator-fingerprint-v2"

    with pytest.raises(RuntimeError, match="Component fingerprint mismatch"):
        experiment.run(store=store)


def test_run_distinguishes_duplicate_case_ids_across_datasets() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator=TracedGenerator(),
            candidate_policy={"num_samples": 1},
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
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4", "dataset": "dataset-1"},
                    )
                ],
            ),
            Dataset(
                dataset_id="dataset-2",
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "3+3"},
                        expected_output={"answer": "6", "dataset": "dataset-2"},
                    )
                ],
            ),
        ],
        seeds=[7],
    )
    store = InMemoryRunStore()

    result = experiment.run(store=store)
    benchmark = store.get_projection(result.run_id, "benchmark_result")
    timeline = store.get_projection(result.run_id, "timeline_view")
    trace_view = store.get_projection(result.run_id, "trace_view")
    stored = store.resume(result.run_id)

    assert result.status.value == "completed"
    assert stored is not None
    assert len(stored.execution_state.case_states) == 2
    assert [case.case_id for case in result.cases] == ["case-1", "case-1"]
    assert {case.dataset_id for case in result.cases} == {"dataset-1", "dataset-2"}
    assert len({case.case_key for case in result.cases}) == 2
    assert {
        (
            case.dataset_id,
            cast(dict[str, JSONValue], case.generated_candidates[0].final_output)[
                "dataset"
            ],
        )
        for case in result.cases
    } == {("dataset-1", "dataset-1"), ("dataset-2", "dataset-2")}
    assert isinstance(benchmark, dict)
    score_rows = cast(list[dict[str, JSONValue]], benchmark["score_rows"])
    assert len(score_rows) == 2
    assert {(row["dataset_id"], row["case_key"]) for row in score_rows} == {
        ("dataset-1", result.cases[0].case_key),
        ("dataset-2", result.cases[1].case_key),
    }
    assert isinstance(timeline, dict)
    timeline_entries = cast(list[dict[str, JSONValue]], timeline["entries"])
    generation_events = [
        entry
        for entry in timeline_entries
        if entry["event_type"] == "generation_completed"
    ]
    assert len(generation_events) == 2
    assert {
        (entry["dataset_id"], entry["case_key"]) for entry in generation_events
    } == {
        ("dataset-1", result.cases[0].case_key),
        ("dataset-2", result.cases[1].case_key),
    }
    assert isinstance(trace_view, dict)
    generation_traces = cast(
        list[dict[str, JSONValue]], trace_view["generation_traces"]
    )
    assert len(generation_traces) == 2
    assert {
        (record["dataset_id"], record["case_key"]) for record in generation_traces
    } == {
        ("dataset-1", result.cases[0].case_key),
        ("dataset-2", result.cases[1].case_key),
    }


def test_compile_keeps_cached_snapshot_until_explicit_rebuild() -> None:
    case = Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[case])],
    )

    compiled = experiment.compile()
    cast(dict[str, JSONValue], case.input)["question"] = "3+3"

    assert experiment.compile() is compiled
    assert experiment.compile().run_id == compiled.run_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "expected_async_name"),
    [
        ("run", {}, "run_async"),
        ("replay", {"stage": "score"}, "replay_async"),
        ("rejudge", {}, "rejudge_async"),
    ],
)
async def test_sync_experiment_entrypoints_reject_running_event_loops(
    method_name: str,
    kwargs: dict[str, object],
    expected_async_name: str,
) -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
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
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
    )
    method = getattr(experiment, method_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match=expected_async_name):
            method(**kwargs)
        gc.collect()

    assert not [
        warning for warning in caught if "was never awaited" in str(warning.message)
    ]


def test_compile_rejects_multiple_parsers() -> None:
    with pytest.raises(ValueError, match="at most one parser"):
        Experiment(
            generation=GenerationConfig(
                generator="builtin/demo_generator",
                candidate_policy={"num_samples": 1},
                reducer="builtin/majority_vote",
            ),
            evaluation=EvaluationConfig(
                metrics=["builtin/exact_match"],
                parsers=["builtin/json_identity", "builtin/json_identity"],
            ),
            storage=StorageConfig(store="memory"),
            datasets=[
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
        )


def test_replay_rejects_unknown_stage() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
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
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        seeds=[7],
    )
    store = InMemoryRunStore()

    experiment.run(store=store)

    with pytest.raises(ValueError, match="Unsupported replay stage"):
        experiment.replay(stage=cast(Any, "unknown"), store=store)
