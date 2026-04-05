from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from themis.core.base import HashableModel
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import GenerationCompletedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.projections import (
    apply_event_to_store_projection_payloads,
    build_initial_store_projection_payloads,
)
from themis.core.snapshot import RunSnapshot


class CountingFingerprintComponent:
    component_id = "component/counting"
    version = "1.0"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        self.calls += 1
        return "fingerprint-counting"


class CountingHashModel(HashableModel):
    component: object
    value: int


def _snapshot() -> RunSnapshot:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
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
        seeds=[7],
    )
    return experiment.compile()


def test_compute_hash_reuses_cached_canonical_fingerprint_work() -> None:
    component = CountingFingerprintComponent()
    model = CountingHashModel(component=component, value=1)

    assert model.compute_hash() == model.compute_hash()
    assert component.calls == 1


def test_projection_updates_reuse_existing_snapshot_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot()
    calls = 0
    original_model_dump = RunSnapshot.model_dump

    def counting_model_dump(self, *args, **kwargs):
        nonlocal calls
        if self is snapshot:
            calls += 1
        return original_model_dump(self, *args, **kwargs)

    monkeypatch.setattr(RunSnapshot, "model_dump", counting_model_dump)

    projections = build_initial_store_projection_payloads(snapshot)
    updated = apply_event_to_store_projection_payloads(
        snapshot,
        projections,
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-7",
            candidate_index=0,
            seed=7,
            result={
                "candidate_id": "case-1-candidate-7",
                "final_output": {"answer": "4"},
            },
        ),
    )

    assert updated["snapshot"] == projections["snapshot"]
    assert calls == 1


def test_profile_script_emits_machine_readable_json() -> None:
    script = Path("scripts/ci/profile_load.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            "20",
            "--samples",
            "2",
            "--judges",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cases"] == 20
    assert payload["samples"] == 2
    assert payload["judges"] == 2
    assert "duration_seconds" in payload
    assert payload["status"] in {"completed", "partial_failure"}
