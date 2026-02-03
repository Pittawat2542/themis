from __future__ import annotations

from themis.api import evaluate
from themis.backends.execution import ExecutionBackend
from themis.backends.storage import LocalFileStorageBackend


class RecordingBackend(ExecutionBackend):
    def __init__(self):
        self.called = False

    def map(self, func, items, *, max_workers=None, timeout=None, **kwargs):
        self.called = True
        for item in items:
            yield func(item)

    def shutdown(self) -> None:
        pass


def test_evaluate_uses_execution_backend(tmp_path):
    backend = RecordingBackend()
    dataset = [{"id": "1", "question": "2+2", "answer": "4"}]

    evaluate(
        dataset,
        model="fake-math-llm",
        prompt="What is {question}?",
        execution_backend=backend,
        storage=tmp_path,
        run_id="execution-backend-test",
        resume=False,
    )

    assert backend.called is True


def test_evaluate_with_local_storage_backend(tmp_path):
    storage_backend = LocalFileStorageBackend(tmp_path)
    dataset = [{"id": "1", "question": "2+2", "answer": "4"}]

    evaluate(
        dataset,
        model="fake-math-llm",
        prompt="What is {question}?",
        storage_backend=storage_backend,
        run_id="storage-backend-test",
        resume=False,
    )

    metadata_path = (
        tmp_path
        / "experiments"
        / "default"
        / "runs"
        / "storage-backend-test"
        / "metadata.json"
    )
    assert metadata_path.exists()
