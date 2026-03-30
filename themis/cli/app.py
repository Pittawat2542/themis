"""Command-line surface for Themis."""

from __future__ import annotations

from typing import Literal, cast

from cyclopts import App

from themis.cli.commands.batch import batch_app
from themis.cli.commands.compare import compare
from themis.cli.commands.export import export_app
from themis.cli.commands.init import init
from themis.cli.commands.quick_eval import quick_eval_app
from themis.cli.commands.reporting import report
from themis.cli.commands.run import estimate, quickcheck, resume, run
from themis.cli.commands.worker import worker_app
from themis.cli.helpers import dump_json, load_experiment
from themis.core.submission import submit_experiment

app = App(name="themis", help="Themis v4 CLI")


@app.command
def submit(*, config: str, mode: Literal["worker-pool", "batch"]) -> int:
    experiment = load_experiment(config)
    normalized_mode = cast(Literal["worker_pool", "batch"], mode.replace("-", "_"))
    manifest = submit_experiment(experiment, config_path=config, mode=normalized_mode)
    print(
        dump_json(
            {
                "run_id": manifest.run_id,
                "status": manifest.status,
                "manifest_path": str(manifest.manifest_path),
                "mode": manifest.mode,
            }
        )
    )
    return 0


app.command(run)
app.command(resume)
app.command(estimate)
app.command(report)
app.command(quickcheck)
app.command(compare)
app.command(init)
app.command(quick_eval_app)
app.command(export_app)
app.command(worker_app)
app.command(batch_app)
