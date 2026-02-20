from __future__ import annotations

import themis
from themis.backends.execution import ExecutionBackend


class RecordingBackend(ExecutionBackend):
    def __init__(self):
        self.called = False
        self.shutdown_called = False

    def map(self, func, items, *, max_workers=None, timeout=None, **kwargs):
        self.called = True
        for item in items:
            yield func(item)

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_evaluate_uses_execution_backend(tmp_path):
    backend = RecordingBackend()
    themis.evaluate(
        [{"id": "1", "question": "2+2", "answer": "4"}],
        model="fake:fake-math-llm",
        prompt="Solve: {question}",
        metrics=["response_length"],
        temperature=0.0,
        run_id="backend-test",
        execution_backend=backend,
        workers=2,
        storage=str(tmp_path),
    )

    assert backend.called is True
    assert backend.shutdown_called is False
