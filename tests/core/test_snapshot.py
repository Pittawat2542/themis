from __future__ import annotations

import json
from pathlib import Path

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset


class DummyGenerator:
    component_id = "generator/demo"
    version = "1.0"

    def __init__(self, fingerprint_value: str) -> None:
        self.fingerprint_value = fingerprint_value

    def fingerprint(self) -> str:
        return self.fingerprint_value


class DummyReducer:
    component_id = "reducer/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-fingerprint"


class DummyParser:
    component_id = "parser/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-fingerprint"


class DummyMetric:
    component_id = "metric/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-fingerprint"


def _experiment(*, revision: str = "r1", generator: DummyGenerator | None = None) -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator=generator or DummyGenerator("generator-fingerprint"),
            candidate_policy={"num_samples": 2},
            reducer=DummyReducer(),
        ),
        evaluation=EvaluationConfig(
            metrics=[DummyMetric()],
            parsers=[DummyParser()],
            judge_config={"panel_size": 1},
        ),
        storage=StorageConfig(store="sqlite", parameters={"path": "runs/themis.sqlite3"}),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
                revision=revision,
                metadata={"owner": "tests"},
            )
        ],
        seeds=[7, 11],
        environment_metadata={"env": "test"},
        themis_version="4.0.0a0",
        python_version="3.12.9",
        platform="macos",
    )


def test_run_id_uses_identity_fields_only() -> None:
    compiled = _experiment().compile()
    changed_provenance = compiled.model_copy(
        update={
            "provenance": compiled.provenance.model_copy(
                update={"platform": "linux", "environment_metadata": {"env": "prod"}}
            )
        }
    )

    assert compiled.run_id == changed_provenance.run_id


def test_identity_changes_alter_run_id() -> None:
    first = _experiment(revision="r1").compile()
    second = _experiment(revision="r2").compile()

    assert first.run_id != second.run_id


def test_component_fingerprints_are_frozen_at_compile_time() -> None:
    generator = DummyGenerator("fingerprint-before")
    compiled = _experiment(generator=generator).compile()

    generator.fingerprint_value = "fingerprint-after"

    assert compiled.component_refs.generator.fingerprint == "fingerprint-before"


def test_snapshot_serialization_matches_golden_file() -> None:
    compiled = _experiment().compile()
    golden_path = Path("tests/core/golden/run_snapshot_minimal.json")

    expected = json.loads(golden_path.read_text())

    assert compiled.model_dump(mode="json") == expected
