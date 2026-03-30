"""Manifest-backed deferred execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from themis.core.base import FrozenModel
from themis.core.experiment import Experiment
from themis.core.results import RunResult
from themis.core.stores.factory import create_run_store


class SubmissionManifest(FrozenModel):
    run_id: str
    mode: Literal["worker_pool", "batch"]
    config_path: str
    manifest_path: Path
    status: str = "pending"


def submit_experiment(
    experiment: Experiment,
    *,
    config_path: str,
    mode: Literal["worker_pool", "batch"],
) -> SubmissionManifest:
    snapshot = experiment.compile()
    store = create_run_store(experiment.storage)
    store.initialize()
    if store.resume(snapshot.run_id) is None:
        store.persist_snapshot(snapshot)

    if mode == "worker_pool":
        root = Path(experiment.runtime.queue_root or "runs/queue")
        for name in ("queued", "claimed", "done"):
            (root / name).mkdir(parents=True, exist_ok=True)
        manifest_path = root / "queued" / f"{snapshot.run_id}.json"
    else:
        root = Path(experiment.runtime.batch_root or "runs/batch")
        for name in ("requests", "completed"):
            (root / name).mkdir(parents=True, exist_ok=True)
        manifest_path = root / "requests" / f"{snapshot.run_id}.json"

    manifest = SubmissionManifest(
        run_id=snapshot.run_id,
        mode=mode,
        config_path=config_path,
        manifest_path=manifest_path,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def run_worker_once(queue_root: str | Path) -> RunResult | None:
    root = Path(queue_root)
    queued = sorted((root / "queued").glob("*.json"))
    if not queued:
        return None

    source = queued[0]
    claimed = root / "claimed" / source.name
    claimed.parent.mkdir(parents=True, exist_ok=True)
    source.rename(claimed)
    manifest = _read_manifest(claimed)
    result = _run_manifest(manifest)
    done = root / "done" / source.name
    done.parent.mkdir(parents=True, exist_ok=True)
    claimed.rename(done)
    return result


def run_batch_request(request: str | Path) -> RunResult:
    request_path = Path(request)
    manifest = _read_manifest(request_path)
    result = _run_manifest(manifest)
    completed = request_path.parent.parent / "completed" / request_path.name
    completed.parent.mkdir(parents=True, exist_ok=True)
    request_path.rename(completed)
    return result


def _read_manifest(path: Path) -> SubmissionManifest:
    return SubmissionManifest.model_validate_json(path.read_text())


def _run_manifest(manifest: SubmissionManifest) -> RunResult:
    experiment = Experiment.from_config(manifest.config_path)
    store = create_run_store(experiment.storage)
    store.initialize()
    return experiment.run(store=store)
