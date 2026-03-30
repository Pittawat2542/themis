from __future__ import annotations

import pytest

from themis import Experiment, RunSnapshot
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset
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
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
                revision="r1",
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version="4.0.0rc1",
        python_version="3.12.9",
        platform="macos",
    )

    snapshot = experiment.compile()

    assert isinstance(snapshot, RunSnapshot)
    assert snapshot.identity.dataset_refs[0].dataset_id == "dataset-1"
    assert snapshot.datasets[0].cases[0].case_id == "case-1"
    assert snapshot.component_refs.generator.component_id == "builtin/demo_generator"
    assert snapshot.component_refs.judge_models[0].component_id == "judge/custom"
    assert snapshot.identity.judge_model_refs[0].fingerprint == "judge-custom-fingerprint"


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
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
            )
        ],
        seeds=[7],
    )

    with pytest.raises(ValueError, match="original store instance"):
        experiment.rejudge()


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
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
            )
        ],
        seeds=[7],
    )
    store = InMemoryRunStore()

    result = experiment.run(store=store)

    assert result.status.value == "completed"
    assert store.resume(experiment.compile().run_id) is not None
