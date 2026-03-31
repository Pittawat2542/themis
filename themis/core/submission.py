"""Manifest-backed deferred execution helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence, cast

from themis.catalog.loaders import load_symbol
from themis.catalog.registry import component_specs, load_component
from themis.core.base import FrozenModel
from themis.core.config import EvaluationConfig, GenerationConfig
from themis.core.config_loading import ExecutionComponentTargets, load_experiment_definition
from themis.core.experiment import Experiment
from themis.core.protocols import (
    CandidateReducer,
    CandidateSelector,
    Generator,
    JudgeModel,
    LLMMetric,
    Parser,
    PureMetric,
    SelectionMetric,
    TraceMetric,
)
from themis.core.results import RunResult
from themis.core.snapshot import RunSnapshot
from themis.core.stores.factory import create_run_store


class SubmissionManifest(FrozenModel):
    run_id: str
    mode: Literal["worker_pool", "batch"]
    config_path: str | None = None
    manifest_path: Path
    snapshot: RunSnapshot
    execution_targets: ExecutionComponentTargets
    status: str = "pending"


def submit_experiment(
    experiment: Experiment,
    *,
    config_path: str,
    mode: Literal["worker_pool", "batch"],
) -> SubmissionManifest:
    snapshot = experiment.compile()
    absolute_config_path = _config_path_for_manifest(experiment, config_path)
    execution_targets = _resolve_execution_targets(experiment, snapshot, config_path=absolute_config_path)
    store = create_run_store(snapshot.provenance.storage)
    store.initialize()
    _persist_or_validate_snapshot(store, snapshot)

    if mode == "worker_pool":
        root = _submission_root(experiment.runtime.queue_root, config_path=absolute_config_path, default_dir="runs/queue")
        for name in ("queued", "claimed", "done"):
            (root / name).mkdir(parents=True, exist_ok=True)
        manifest_path = root / "queued" / f"{snapshot.run_id}.json"
    else:
        root = _submission_root(experiment.runtime.batch_root, config_path=absolute_config_path, default_dir="runs/batch")
        for name in ("requests", "completed"):
            (root / name).mkdir(parents=True, exist_ok=True)
        manifest_path = root / "requests" / f"{snapshot.run_id}.json"

    manifest = SubmissionManifest(
        run_id=snapshot.run_id,
        mode=mode,
        config_path=absolute_config_path,
        manifest_path=manifest_path,
        snapshot=snapshot,
        execution_targets=execution_targets,
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
    payload = json.loads(path.read_text())
    snapshot_payload = payload.get("snapshot")
    if isinstance(snapshot_payload, dict):
        snapshot_payload = dict(snapshot_payload)
        snapshot_payload.pop("run_id", None)
        payload = dict(payload)
        payload["snapshot"] = snapshot_payload
    return SubmissionManifest.model_validate(payload)


def _run_manifest(manifest: SubmissionManifest) -> RunResult:
    store = create_run_store(manifest.snapshot.provenance.storage)
    store.initialize()
    _persist_or_validate_snapshot(store, manifest.snapshot)
    experiment = _experiment_from_manifest(manifest)
    return experiment.run(store=store)


def _experiment_from_manifest(manifest: SubmissionManifest) -> Experiment:
    snapshot = manifest.snapshot
    targets = manifest.execution_targets
    experiment = Experiment(
        generation=GenerationConfig(
            generator=cast(Generator | str, _resolve_execution_target(targets.generator, kind="generator")),
            selector=cast(CandidateSelector | str | None, _resolve_execution_target(targets.selector, kind="selector"))
            if targets.selector is not None
            else None,
            candidate_policy=snapshot.identity.candidate_policy,
            reducer=cast(CandidateReducer | str | None, _resolve_execution_target(targets.reducer, kind="reducer"))
            if targets.reducer is not None
            else None,
        ),
        evaluation=EvaluationConfig(
            metrics=[
                cast(
                    PureMetric | LLMMetric | SelectionMetric | TraceMetric | str,
                    _resolve_execution_target(target, kind="metric"),
                )
                for target in targets.metrics
            ],
            parsers=[cast(Parser | str, _resolve_execution_target(target, kind="parser")) for target in targets.parsers],
            judge_models=[
                cast(JudgeModel | str, _resolve_execution_target(target, kind="judge_model"))
                for target in targets.judge_models
            ],
            judge_config=snapshot.identity.judge_config,
            workflow_overrides=snapshot.identity.workflow_overrides,
        ),
        storage=snapshot.provenance.storage,
        runtime=snapshot.provenance.runtime,
        datasets=snapshot.datasets,
        seeds=snapshot.identity.seeds,
        environment_metadata=snapshot.provenance.environment_metadata,
        themis_version=snapshot.provenance.themis_version,
        python_version=snapshot.provenance.python_version,
        platform=snapshot.provenance.platform,
        git_commit=snapshot.provenance.git_commit,
        dependency_versions=snapshot.provenance.dependency_versions,
        provider_metadata=snapshot.provenance.provider_metadata,
    )
    experiment._compiled_snapshot = snapshot
    return experiment


def _resolve_execution_target(target: str, *, kind: str) -> object:
    if target in component_specs():
        return load_component(target, kind=kind)
    loaded = load_symbol(target)
    if isinstance(loaded, type):
        return loaded()
    return loaded


def _persist_or_validate_snapshot(store, snapshot: RunSnapshot) -> None:
    stored = store.resume(snapshot.run_id)
    if stored is None:
        store.persist_snapshot(snapshot)
        return
    if stored.snapshot != snapshot:
        raise ValueError(f"Stored snapshot does not match submitted manifest for run_id={snapshot.run_id}")


def _submission_root(runtime_root: str | None, *, config_path: str | None, default_dir: str) -> Path:
    if runtime_root:
        return Path(runtime_root)
    if config_path is not None:
        return Path(config_path).parent / default_dir
    return Path(default_dir)


def _config_path_for_manifest(experiment: Experiment, config_path: str) -> str | None:
    if experiment._config_metadata is not None:
        return str(experiment._config_metadata.config_path)
    path = Path(config_path).expanduser()
    return str(path.resolve()) if path.exists() else None


def _resolve_execution_targets(
    experiment: Experiment,
    snapshot: RunSnapshot,
    *,
    config_path: str | None,
) -> ExecutionComponentTargets:
    config_targets = experiment._config_metadata.component_targets if experiment._config_metadata is not None else None
    if config_targets is None and config_path is not None:
        path = Path(config_path)
        if path.exists():
            loaded = load_experiment_definition(path)
            if Experiment.model_validate(loaded.payload).compile().run_id == snapshot.run_id:
                config_targets = loaded.metadata.component_targets

    generator = _select_target(experiment.generation.generator, config_targets.generator if config_targets else None)
    selector = _select_target(experiment.generation.selector, config_targets.selector if config_targets else None)
    reducer = _select_target(experiment.generation.reducer, config_targets.reducer if config_targets else None)
    parsers = _select_target_list(
        experiment.evaluation.parsers,
        config_targets.parsers if config_targets is not None else None,
    )
    metrics = _select_target_list(
        experiment.evaluation.metrics,
        config_targets.metrics if config_targets is not None else None,
    )
    judge_models = _select_target_list(
        experiment.evaluation.judge_models,
        config_targets.judge_models if config_targets is not None else None,
    )
    if generator is None or any(target is None for target in parsers + metrics + judge_models):
        raise ValueError(
            "submit_experiment only supports builtin components or importable config symbols; "
            "define custom components in config via module:symbol paths."
        )
    if experiment.generation.selector is not None and selector is None:
        raise ValueError(
            "submit_experiment only supports builtin components or importable config symbols; "
            "define custom components in config via module:symbol paths."
        )
    if experiment.generation.reducer is not None and reducer is None:
        raise ValueError(
            "submit_experiment only supports builtin components or importable config symbols; "
            "define custom components in config via module:symbol paths."
        )
    return ExecutionComponentTargets(
        generator=generator,
        selector=selector,
        reducer=reducer,
        parsers=[target for target in parsers if target is not None],
        metrics=[target for target in metrics if target is not None],
        judge_models=[target for target in judge_models if target is not None],
    )


def _select_target(value: object | None, fallback: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return fallback


def _select_target_list(values: Sequence[object], fallback: list[str] | None) -> list[str | None]:
    fallback_values = list(fallback or [])
    if fallback_values and len(fallback_values) != len(values):
        raise ValueError("Config component target metadata does not match experiment component counts")
    targets: list[str | None] = []
    for index, value in enumerate(values):
        targets.append(value if isinstance(value, str) else (fallback_values[index] if fallback_values else None))
    return targets
