"""Execution backend interfaces for deferred graph execution."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from themis.core.base import FrozenModel, JSONValue


class ExecutionRequest(FrozenModel):
    """One submitted execution request."""

    run_id: str
    payload: dict[str, JSONValue] = Field(default_factory=dict)


class ExecutionBackend:
    """Queue-like execution backend contract."""

    def submit(self, request: ExecutionRequest) -> None:
        raise NotImplementedError

    def claim_next(self) -> ExecutionRequest | None:
        raise NotImplementedError

    def complete(self, run_id: str, result: dict[str, JSONValue]) -> None:
        raise NotImplementedError

    def completed(self, run_id: str) -> dict[str, JSONValue] | None:
        raise NotImplementedError


class InMemoryExecutionBackend(ExecutionBackend):
    """In-process execution backend for tests and local orchestration."""

    def __init__(self) -> None:
        self._queued: list[ExecutionRequest] = []
        self._claimed: dict[str, ExecutionRequest] = {}
        self._completed: dict[str, dict[str, JSONValue]] = {}

    def submit(self, request: ExecutionRequest) -> None:
        self._queued.append(request)

    def claim_next(self) -> ExecutionRequest | None:
        if not self._queued:
            return None
        request = self._queued.pop(0)
        self._claimed[request.run_id] = request
        return request

    def complete(self, run_id: str, result: dict[str, JSONValue]) -> None:
        self._claimed.pop(run_id, None)
        self._completed[run_id] = dict(result)

    def completed(self, run_id: str) -> dict[str, JSONValue] | None:
        result = self._completed.get(run_id)
        return None if result is None else dict(result)


class FilesystemExecutionBackend(ExecutionBackend):
    """Filesystem queue backend with queued, claimed, and completed directories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for name in ("queued", "claimed", "completed"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def submit(self, request: ExecutionRequest) -> None:
        self._request_path("queued", request.run_id).write_text(
            request.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def claim_next(self) -> ExecutionRequest | None:
        queued = sorted((self.root / "queued").glob("*.json"))
        if not queued:
            return None
        queued_path = queued[0]
        claimed_path = self.root / "claimed" / queued_path.name
        queued_path.rename(claimed_path)
        return ExecutionRequest.model_validate_json(
            claimed_path.read_text(encoding="utf-8")
        )

    def complete(self, run_id: str, result: dict[str, JSONValue]) -> None:
        claimed_path = self._request_path("claimed", run_id)
        completed_path = self._request_path("completed", run_id)
        if claimed_path.exists():
            claimed_path.unlink()
        completed_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def completed(self, run_id: str) -> dict[str, JSONValue] | None:
        completed_path = self._request_path("completed", run_id)
        if not completed_path.exists():
            return None
        payload = json.loads(completed_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Completed payload is not an object for run_id={run_id}")
        return payload

    def _request_path(self, state: str, run_id: str) -> Path:
        return self.root / state / f"{run_id}.json"
