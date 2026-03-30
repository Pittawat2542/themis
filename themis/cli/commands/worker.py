"""Worker-pool CLI commands."""

from __future__ import annotations

from cyclopts import App

from themis.cli.helpers import dump_json
from themis.core.submission import run_worker_once

worker_app = App(name="worker", help="Worker-pool operations.")


@worker_app.command
def run(*, queue_root: str = "runs/queue", once: bool = False) -> int:
    del once
    result = run_worker_once(queue_root)
    if result is None:
        print(dump_json({"status": "idle"}))
        return 0
    print(
        dump_json(
            {
                "run_id": result.run_id,
                "status": result.status.value,
            }
        )
    )
    return 0
