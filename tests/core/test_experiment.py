from __future__ import annotations

from themis import Experiment, RunSnapshot
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def test_experiment_compile_returns_snapshot() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
            judge_config={"panel_size": 1},
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
        themis_version="4.0.0a0",
        python_version="3.12.9",
        platform="macos",
    )

    snapshot = experiment.compile()

    assert isinstance(snapshot, RunSnapshot)
    assert snapshot.identity.dataset_refs[0].dataset_id == "dataset-1"
    assert snapshot.datasets[0].cases[0].case_id == "case-1"
    assert snapshot.component_refs.generator.component_id == "generator/demo"
