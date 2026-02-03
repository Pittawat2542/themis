from __future__ import annotations

import pytest

from themis.evaluation import extractors, metrics, pipeline as evaluation_pipeline
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec


def test_experiment_session_runs_with_list_dataset(tmp_path):
    pipeline = evaluation_pipeline.EvaluationPipeline(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
    )

    spec = ExperimentSpec(
        dataset=[{"id": "1", "question": "2+2", "answer": "4"}],
        prompt="Solve: {question}",
        model="fake:fake-math-llm",
        sampling={"temperature": 0.0},
        pipeline=pipeline,
        run_id="session-test",
    )

    session = ExperimentSession()
    report = session.run(
        spec,
        execution=ExecutionSpec(workers=1),
        storage=StorageSpec(path=tmp_path),
    )

    assert report.metadata["total_samples"] == 1
    assert "ResponseLength" in report.evaluation_report.metrics


def test_experiment_session_rejects_bad_pipeline():
    spec = ExperimentSpec(
        dataset=[{"id": "1"}],
        prompt="Q",
        model="fake:fake-math-llm",
        pipeline=object(),
    )

    session = ExperimentSession()

    with pytest.raises(TypeError, match="EvaluationPipelineContract"):
        session.run(spec)
