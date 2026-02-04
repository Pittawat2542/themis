from __future__ import annotations

from pathlib import Path

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline as evaluation_pipeline
from themis.session import ExperimentSession
from themis.specs import ExperimentSpec, StorageSpec


class FakeStorage:
    def __init__(self):
        self.start_called = False

    def run_metadata_exists(self, run_id: str) -> bool:
        return False

    def start_run(
        self,
        run_id: str,
        experiment_id: str = "default",
        config: dict | None = None,
    ) -> None:
        self.start_called = True

    def cache_dataset(self, run_id: str, dataset):
        pass

    def load_cached_records(self, run_id: str):
        return {}

    def load_cached_evaluations(self, run_id: str, eval_id: str = "default", evaluation_config=None):
        return {}

    def append_record(self, run_id: str, record, cache_key: str | None = None):
        pass

    def append_evaluation(self, run_id: str, record, evaluation, eval_id: str = "default", evaluation_config=None):
        pass

    def get_run_path(self, run_id: str):
        return None


def test_session_uses_storage_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

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
        run_id="session-storage-test",
    )

    backend = FakeStorage()
    session = ExperimentSession()

    session.run(spec, storage=StorageSpec(backend=backend))

    assert backend.start_called is True
    assert Path("None").exists() is False
