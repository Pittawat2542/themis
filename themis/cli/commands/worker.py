"""Worker-pool CLI commands."""

from __future__ import annotations

import os

from cyclopts import App

from themis.cli.helpers import dump_json
from themis.core.submission import run_worker_once

worker_app = App(name="worker", help="Worker-pool operations.")


@worker_app.command
def run(
    *,
    definition_root: list[str],
    queue_root: str = "runs/queue",
    worker_id: str | None = None,
    lease_seconds: int = 300,
    signing_key_env: str | None = None,
    require_signature: bool = False,
) -> int:
    signing_key = os.environ.get(signing_key_env) if signing_key_env else None
    result = run_worker_once(
        queue_root,
        definition_roots=definition_root,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        signing_key=signing_key,
        require_signature=require_signature,
    )
    if result is None:
        print(dump_json("worker.run", {"status": "idle"}))
        return 0
    print(
        dump_json(
            "worker.run",
            {
                "run_id": result.run_id,
                "status": result.status.value,
            },
        )
    )
    return 0
