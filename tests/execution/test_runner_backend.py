from __future__ import annotations

from themis.backends.execution import ExecutionBackend
from themis.evaluation import extractors, metrics, pipeline as evaluation_pipeline
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec


class RecordingBackend(ExecutionBackend):
    def __init__(self):
        self.called = False

    def map(self, func, items, *, max_workers=None, timeout=None, **kwargs):
        self.called = True
        for item in items:
            yield func(item)

    def shutdown(self) -> None:
        pass


def test_session_uses_execution_backend(tmp_path):
    backend = RecordingBackend()
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
        run_id="session-backend-test",
    )

    session = ExperimentSession()
    session.run(
        spec,
        execution=ExecutionSpec(backend=backend, workers=2),
        storage=StorageSpec(path=tmp_path),
    )

    assert backend.called is True
