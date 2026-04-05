"""Batch CLI commands."""

from __future__ import annotations

from cyclopts import App

from themis.cli.helpers import dump_json
from themis.core.submission import run_batch_request

batch_app = App(name="batch", help="Batch execution operations.")


@batch_app.command
def run(*, request: str) -> int:
    result = run_batch_request(request)
    print(
        dump_json(
            {
                "run_id": result.run_id,
                "status": result.status.value,
            }
        )
    )
    return 0
