from __future__ import annotations

import pytest

from themis import Case, Dataset, Evaluation, Experiment, Generation, RunOptions
from themis.presets import (
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
        generation=Generation(
            generator="builtin/demo_generator",
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=["builtin/exact_match"],
            parser="builtin/json_identity",
        ),
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
    application = apply_preset(experiment, "runtime/local-careful")

    assert application.experiment is not experiment
    assert application.options.max_concurrency != RunOptions().max_concurrency
    assert application.experiment.compile().run_id == experiment.compile().run_id
    assert application.preset_ids == ["runtime/local-careful"]
    assert application.experiment.metadata["themis.preset.runtime/local-careful"] == (
        "true"
    )


def test_session_preset_changes_logical_identity() -> None:
    experiment = _experiment()
    application = apply_preset(experiment, "candidate/best-of-n")

    assert application.experiment.generation.samples == 2
    assert application.experiment.compile().run_id != experiment.compile().run_id


def test_unknown_preset_error_includes_suggestion() -> None:
    with pytest.raises(ValueError, match="runtime/local-fast"):
        get_preset("runtime/local-fst")
