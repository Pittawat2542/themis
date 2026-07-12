from __future__ import annotations

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.presets import (
    EvaluationPreset,
    ExperimentPreset,
    RuntimePreset,
    GenerationPreset,
    apply_preset,
    get_preset,
    list_presets,
)


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
    )


def test_presets_are_discoverable_by_kind() -> None:
    assert "runtime/local-fast" in list_presets(kind="runtime")
    assert "candidate/best-of-n" in list_presets(kind="generation")
    assert "judge/demo-rubric" in list_presets(kind="evaluation")

    assert isinstance(get_preset("runtime/local-fast"), RuntimePreset)
    assert isinstance(get_preset("candidate/best-of-n"), GenerationPreset)
    assert isinstance(get_preset("judge/demo-rubric"), EvaluationPreset)
    assert isinstance(get_preset("baseline/demo"), ExperimentPreset)


def test_runtime_only_preset_changes_provenance_not_run_identity() -> None:
    experiment = _experiment()
    updated = apply_preset(experiment, "runtime/local-careful")

    assert updated is not experiment
    assert (
        updated.runtime.max_concurrent_tasks != experiment.runtime.max_concurrent_tasks
    )
    assert updated.compile().run_id == experiment.compile().run_id
    assert updated.environment_metadata["themis.preset.runtime/local-careful"] == "true"


def test_session_preset_changes_logical_identity() -> None:
    experiment = _experiment()
    updated = apply_preset(experiment, "candidate/best-of-n")

    assert updated.generation.candidate_policy["num_samples"] == 2
    assert updated.compile().run_id != experiment.compile().run_id


def test_unknown_preset_error_includes_suggestion() -> None:
    with pytest.raises(ValueError, match="runtime/local-fast"):
        get_preset("runtime/local-fst")
