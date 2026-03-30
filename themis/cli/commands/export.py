"""Export CLI commands."""

from __future__ import annotations

from cyclopts import App

from themis.cli.helpers import dump_json, initialize_store, load_experiment
from themis.core.bundles import export_evaluation_bundle, export_generation_bundle

export_app = App(name="export", help="Export stored run artifacts.")


@export_app.command
def generation(*, config: str) -> int:
    experiment = load_experiment(config)
    bundle = export_generation_bundle(initialize_store(experiment), experiment.compile().run_id)
    print(dump_json(bundle.model_dump(mode="json")))
    return 0


@export_app.command
def evaluation(*, config: str) -> int:
    experiment = load_experiment(config)
    bundle = export_evaluation_bundle(initialize_store(experiment), experiment.compile().run_id)
    print(dump_json(bundle.model_dump(mode="json")))
    return 0
