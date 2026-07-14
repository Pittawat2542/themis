"""Shared CLI helpers."""

from __future__ import annotations

from datetime import datetime
import json

from themis.core.experiment import Experiment
from themis.core.inspection import resolve_run_id
from themis.core.read_models import BenchmarkResult
from themis.core.registry import RunQuery
from themis.core.store import RunStore
from themis.core.stores.factory import create_run_store
from themis.launcher import _load_runtime_experiment


def dump_json(command: str, payload: object) -> str:
    """Render one versioned machine-readable CLI response."""

    return json.dumps(
        {"schema_version": "1", "command": command, "data": payload},
        indent=2,
        sort_keys=True,
    )


def load_experiment(config: str, *, overrides: list[str] | None = None) -> Experiment:
    """Load an experiment definition from a config file path."""

    return _load_runtime_experiment(config, overrides=overrides)


def initialize_store(experiment: Experiment) -> RunStore:
    """Create and initialize the configured store for an experiment."""

    store = create_run_store(experiment.storage)
    store.initialize()
    return store


def build_run_query(
    *,
    run_id: str | None = None,
    dataset_source_id: str | None = None,
    dataset_fingerprint: str | None = None,
    metric_id: str | None = None,
    tags: list[str] | None = None,
    baseline_label: str | None = None,
    lineage_parent_run_id: str | None = None,
    status: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> RunQuery:
    """Build a registry query from CLI parameters."""

    return RunQuery(
        run_id=run_id,
        dataset_source_id=dataset_source_id,
        dataset_fingerprint=dataset_fingerprint,
        metric_id=metric_id,
        tags=list(tags or []),
        baseline_label=baseline_label,
        lineage_parent_run_id=lineage_parent_run_id,
        status=status,
        created_after=_maybe_parse_datetime(created_after),
        created_before=_maybe_parse_datetime(created_before),
        updated_after=_maybe_parse_datetime(updated_after),
        updated_before=_maybe_parse_datetime(updated_before),
    )


def resolve_persisted_run_id(
    store: RunStore,
    *,
    run_id: str | None = None,
    baseline_label: str | None = None,
    query: RunQuery | None = None,
) -> str:
    """Resolve a persisted run id by explicit id or registry query."""

    return resolve_run_id(
        store, run_id=run_id, baseline_label=baseline_label, query=query
    )


def load_benchmark_result(store: RunStore, run_id: str) -> BenchmarkResult:
    """Load a benchmark-result projection from a configured store."""

    projection = store.get_projection(run_id, "benchmark_result")
    if not isinstance(projection, dict):
        raise ValueError(f"Benchmark projection unavailable for run_id={run_id}")
    return BenchmarkResult.model_validate(projection)


def _maybe_parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)
