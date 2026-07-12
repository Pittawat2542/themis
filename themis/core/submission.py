"""Manifest-backed deferred execution for Python-defined experiments."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from themis.core.base import FrozenModel
from themis.core.experiment import Experiment
from themis.core.planner import Planner
from themis.core.results import ExecutionResourcePlan, RunResult
from themis.core.snapshot import RunSnapshot
from themis.core.stores.factory import create_run_store


class SubmissionManifest(FrozenModel):
    """Portable request that points to a reviewed launcher and frozen snapshot."""

    schema_version: str = "2"
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    mode: Literal["worker_pool", "batch"]
    config_path: str
    definition_path: str
    definition_digest: str
    manifest_path: Path
    snapshot: RunSnapshot
    status: str = "pending"
    suite_id: str | None = None
    preset_ids: list[str] = Field(default_factory=list)
    resource_plan: ExecutionResourcePlan | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempt_count: int = 0
    worker_id: str | None = None
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    lease_token: str | None = None
    signature: str | None = None
    failure: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_snapshot_run_id(cls, payload: object) -> object:
        if not isinstance(payload, dict):
            return payload
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict) or "run_id" not in snapshot:
            return payload
        normalized = dict(payload)
        normalized_snapshot = dict(snapshot)
        normalized_snapshot.pop("run_id", None)
        normalized["snapshot"] = normalized_snapshot
        return normalized


def submit_experiment(
    experiment: Experiment,
    *,
    config_path: str,
    mode: Literal["worker_pool", "batch"],
    suite_id: str | None = None,
    preset_ids: Sequence[str] = (),
    tags: Sequence[str] = (),
    signing_key: str | bytes | None = None,
) -> SubmissionManifest:
    """Freeze and enqueue an experiment loaded through a v5 launcher."""

    launcher_path = Path(config_path).expanduser().resolve()
    if not launcher_path.is_file():
        raise ValueError("Deferred execution requires an accessible launcher config")
    snapshot = experiment.compile()
    from themis.launcher import resolve_definition_path

    definition_path = resolve_definition_path(launcher_path)
    store = create_run_store(snapshot.provenance.storage)
    store.initialize()
    _persist_or_validate_snapshot(store, snapshot)

    runtime_root = (
        experiment.runtime.queue_root
        if mode == "worker_pool"
        else experiment.runtime.batch_root
    )
    default_dir = "runs/queue" if mode == "worker_pool" else "runs/batch"
    root = Path(runtime_root) if runtime_root else launcher_path.parent / default_dir
    pending_dir = "queued" if mode == "worker_pool" else "requests"
    completed_dir = "done" if mode == "worker_pool" else "completed"
    for name in (pending_dir, completed_dir, "claimed", "failed"):
        (root / name).mkdir(parents=True, exist_ok=True)
    manifest_path = root / pending_dir / f"{snapshot.run_id}.json"

    manifest = SubmissionManifest(
        run_id=snapshot.run_id,
        mode=mode,
        config_path=str(launcher_path),
        definition_path=str(definition_path),
        definition_digest=_file_digest(definition_path),
        manifest_path=manifest_path,
        snapshot=snapshot,
        suite_id=suite_id,
        preset_ids=list(preset_ids),
        resource_plan=Planner().resource_plan(snapshot, snapshot.provenance.runtime),
        tags=list(tags),
    )
    if signing_key is not None:
        manifest = manifest.model_copy(
            update={"signature": _manifest_signature(manifest, signing_key)}
        )
    _write_manifest(manifest_path, manifest)
    return manifest


class SubmissionSecurityError(ValueError):
    """Manifest failed validation before executable code import."""


def run_worker_once(
    queue_root: str | Path,
    *,
    definition_roots: Sequence[str | Path],
    worker_id: str | None = None,
    lease_seconds: int = 300,
    max_attempts: int = 3,
    signing_key: str | bytes | None = None,
    require_signature: bool = False,
) -> RunResult | None:
    root = Path(queue_root)
    _validate_queue_permissions(root, signatures_required=require_signature)
    _recover_expired_claims(root, max_attempts=max_attempts)
    queued = sorted((root / "queued").glob("*.json"))
    if not queued:
        return None
    source = queued[0]
    claimed = root / "claimed" / source.name
    claimed.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.rename(claimed)
    except FileNotFoundError:
        return None
    now = datetime.now(UTC)
    manifest = _read_manifest(claimed).model_copy(
        update={
            "status": "claimed",
            "attempt_count": _read_manifest(claimed).attempt_count + 1,
            "worker_id": worker_id or f"worker-{os.getpid()}",
            "claimed_at": now,
            "lease_expires_at": now + timedelta(seconds=max(1, lease_seconds)),
            "lease_token": secrets.token_hex(16),
            "failure": None,
        }
    )
    _write_manifest(claimed, manifest)
    stop_renewal = threading.Event()
    renewer = _start_lease_renewer(
        claimed, manifest.lease_token or "", lease_seconds, stop_renewal
    )
    try:
        result = _run_manifest(
            manifest,
            definition_roots=definition_roots,
            signing_key=signing_key,
            require_signature=require_signature,
        )
        stop_renewal.set()
        renewer.join(timeout=1.0)
        current = _read_manifest(claimed)
        if current.lease_token != manifest.lease_token:
            raise RuntimeError("Worker no longer owns the active manifest lease")
        done = root / "done" / source.name
        done.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(done, current.model_copy(update={"status": "completed"}))
        claimed.unlink()
        return result
    except Exception as exc:
        current = _read_manifest(claimed)
        failed = isinstance(exc, SubmissionSecurityError) or (
            current.attempt_count >= max_attempts
        )
        destination = root / ("failed" if failed else "queued") / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(
            destination,
            current.model_copy(
                update={
                    "status": "failed" if failed else "pending",
                    "worker_id": None,
                    "claimed_at": None,
                    "lease_expires_at": None,
                    "lease_token": None,
                    "failure": {
                        "exception_class": type(exc).__qualname__,
                        "message": str(exc),
                    },
                }
            ),
        )
        claimed.unlink(missing_ok=True)
        raise
    finally:
        stop_renewal.set()
        renewer.join(timeout=1.0)


def run_batch_request(
    request: str | Path,
    *,
    definition_roots: Sequence[str | Path],
    signing_key: str | bytes | None = None,
    require_signature: bool = False,
) -> RunResult:
    request_path = Path(request)
    result = _run_manifest(
        _read_manifest(request_path),
        definition_roots=definition_roots,
        signing_key=signing_key,
        require_signature=require_signature,
    )
    completed = request_path.parent.parent / "completed" / request_path.name
    completed.parent.mkdir(parents=True, exist_ok=True)
    request_path.rename(completed)
    return result


def _read_manifest(path: Path) -> SubmissionManifest:
    return SubmissionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _run_manifest(
    manifest: SubmissionManifest,
    *,
    definition_roots: Sequence[str | Path],
    signing_key: str | bytes | None,
    require_signature: bool,
) -> RunResult:
    from themis.launcher import load_core_experiment

    _validate_manifest(
        manifest,
        definition_roots=definition_roots,
        signing_key=signing_key,
        require_signature=require_signature,
    )

    experiment = load_core_experiment(manifest.config_path)
    compiled = experiment.compile()
    if compiled.run_id != manifest.run_id:
        raise ValueError(
            "Launcher experiment identity no longer matches the submitted snapshot"
        )
    store = create_run_store(manifest.snapshot.provenance.storage)
    store.initialize()
    _persist_or_validate_snapshot(store, manifest.snapshot)
    experiment._compiled_snapshot = manifest.snapshot
    return experiment.run(store=store)


def _validate_manifest(
    manifest: SubmissionManifest,
    *,
    definition_roots: Sequence[str | Path],
    signing_key: str | bytes | None,
    require_signature: bool,
) -> None:
    if require_signature and manifest.signature is None:
        raise SubmissionSecurityError("Manifest signature is required")
    if manifest.signature is not None:
        if signing_key is None or not hmac.compare_digest(
            manifest.signature, _manifest_signature(manifest, signing_key)
        ):
            raise SubmissionSecurityError("Invalid manifest signature")
    definition_path = Path(manifest.definition_path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in definition_roots]
    if not roots or not any(
        definition_path == root or root in definition_path.parents for root in roots
    ):
        raise SubmissionSecurityError(
            "Definition path is outside configured definition roots"
        )
    if not definition_path.is_file():
        raise SubmissionSecurityError("Definition path no longer exists")
    if _file_digest(definition_path) != manifest.definition_digest:
        raise SubmissionSecurityError("Definition digest no longer matches manifest")


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _manifest_signature(manifest: SubmissionManifest, key: str | bytes) -> str:
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    payload = manifest.model_dump(
        mode="json",
        exclude={
            "attempt_count",
            "claimed_at",
            "failure",
            "lease_expires_at",
            "lease_token",
            "signature",
            "status",
            "worker_id",
        },
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hmac.new(key_bytes, encoded, hashlib.sha256).hexdigest()


def _write_manifest(path: Path, manifest: SubmissionManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _recover_expired_claims(root: Path, *, max_attempts: int) -> None:
    now = datetime.now(UTC)
    for claimed in sorted((root / "claimed").glob("*.json")):
        manifest = _read_manifest(claimed)
        if manifest.lease_expires_at is None or manifest.lease_expires_at > now:
            continue
        exhausted = manifest.attempt_count >= max_attempts
        destination = root / ("failed" if exhausted else "queued") / claimed.name
        _write_manifest(
            destination,
            manifest.model_copy(
                update={
                    "status": "failed" if exhausted else "pending",
                    "worker_id": None,
                    "claimed_at": None,
                    "lease_expires_at": None,
                    "lease_token": None,
                    "failure": {
                        "exception_class": "WorkerLeaseExpired",
                        "message": "Worker lease expired before completion",
                    },
                }
            ),
        )
        claimed.unlink(missing_ok=True)


def _start_lease_renewer(
    path: Path,
    lease_token: str,
    lease_seconds: int,
    stop: threading.Event,
) -> threading.Thread:
    interval = max(1.0, lease_seconds / 3)

    def renew() -> None:
        while not stop.wait(interval):
            if not path.is_file():
                return
            manifest = _read_manifest(path)
            if manifest.lease_token != lease_token:
                return
            _write_manifest(
                path,
                manifest.model_copy(
                    update={
                        "lease_expires_at": datetime.now(UTC)
                        + timedelta(seconds=max(1, lease_seconds))
                    }
                ),
            )

    thread = threading.Thread(target=renew, daemon=True, name="themis-lease-renewer")
    thread.start()
    return thread


def _validate_queue_permissions(root: Path, *, signatures_required: bool) -> None:
    if os.name != "posix" or signatures_required:
        return
    if root.stat().st_mode & 0o022:
        raise PermissionError(
            "Unsigned worker queues must not be group- or world-writable"
        )


def _persist_or_validate_snapshot(store, snapshot: RunSnapshot) -> None:
    stored = store.resume(snapshot.run_id)
    if stored is None:
        store.persist_snapshot(snapshot)
        return
    if stored.snapshot != snapshot:
        raise ValueError(
            f"Stored snapshot does not match submitted manifest for run_id={snapshot.run_id}"
        )
