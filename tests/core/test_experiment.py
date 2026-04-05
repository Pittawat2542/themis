from __future__ import annotations

import pytest
from typing import Any, cast

from themis import Experiment, RunSnapshot
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import GenerateContext
from themis.core.models import Case, Dataset, GenerationResult
from themis.core.stores.memory import InMemoryRunStore
from themis.core.workflows import JudgeResponse


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
        themis_version="4.0.0",
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
