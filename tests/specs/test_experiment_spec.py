from __future__ import annotations

import pytest

from themis.specs import ExperimentSpec


def test_experiment_spec_minimal():
    spec = ExperimentSpec(
        dataset=[],
        prompt="Solve: {question}",
        model="litellm:gpt-4",
        sampling={"temperature": 0.0},
        pipeline=object(),
        run_id="run-1",
    )

    assert spec.prompt == "Solve: {question}"
    assert spec.model == "litellm:gpt-4"
    assert spec.run_id == "run-1"
    assert spec.sampling["temperature"] == 0.0


def test_experiment_spec_requires_prompt():
    with pytest.raises(ValueError, match="prompt"):
        ExperimentSpec(dataset=[], prompt="", model="m", pipeline=object())


def test_experiment_spec_requires_model():
    with pytest.raises(ValueError, match="model"):
        ExperimentSpec(dataset=[], prompt="Q", model="", pipeline=object())


def test_experiment_spec_requires_pipeline():
    with pytest.raises(ValueError, match="pipeline"):
        ExperimentSpec(dataset=[], prompt="Q", model="m", pipeline=None)
