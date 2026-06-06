from __future__ import annotations

from pathlib import Path

from themis.core.execution_backends import (
    ExecutionRequest,
    FilesystemExecutionBackend,
    InMemoryExecutionBackend,
)


def test_in_memory_execution_backend_claims_submitted_requests_once() -> None:
    backend = InMemoryExecutionBackend()
    request = ExecutionRequest(run_id="run-1", payload={"config_path": "eval.yaml"})

    backend.submit(request)

    assert backend.claim_next() == request
    assert backend.claim_next() is None
    backend.complete("run-1", {"status": "completed"})
    assert backend.completed("run-1") == {"status": "completed"}


def test_filesystem_execution_backend_uses_queue_layout(tmp_path: Path) -> None:
    backend = FilesystemExecutionBackend(tmp_path)
    request = ExecutionRequest(run_id="run-1", payload={"config_path": "eval.yaml"})

    backend.submit(request)
    claimed = backend.claim_next()

    assert claimed == request
    assert not (tmp_path / "queued" / "run-1.json").exists()
    assert (tmp_path / "claimed" / "run-1.json").exists()

    backend.complete("run-1", {"status": "completed"})

    assert not (tmp_path / "claimed" / "run-1.json").exists()
    assert backend.completed("run-1") == {"status": "completed"}
