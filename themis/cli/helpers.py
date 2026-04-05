"""Shared CLI helpers."""

from __future__ import annotations

import json

from themis.core.experiment import Experiment
from themis.core.read_models import BenchmarkResult
from themis.core.store import RunStore
from themis.core.stores.factory import create_run_store


def dump_json(payload: object) -> str:
    """Render a JSON payload with stable formatting for CLI output."""

    return json.dumps(payload, indent=2, sort_keys=True)


def load_experiment(config: str, *, overrides: list[str] | None = None) -> Experiment:
    """Load an experiment definition from a config file path."""

    return Experiment.from_config(config, overrides=overrides)


def initialize_store(experiment: Experiment) -> RunStore:
    """Create and initialize the configured store for an experiment."""

    store = create_run_store(experiment.storage)
    store.initialize()
    return store


def load_benchmark_result(store: RunStore, run_id: str) -> BenchmarkResult:
    """Load a benchmark-result projection from a configured store."""

    projection = store.get_projection(run_id, "benchmark_result")
    if not isinstance(projection, dict):
        raise ValueError(f"Benchmark projection unavailable for run_id={run_id}")
    return BenchmarkResult.model_validate(projection)
