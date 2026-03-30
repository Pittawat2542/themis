"""Shared CLI helpers."""

from __future__ import annotations

import json

from themis.core.experiment import Experiment
from themis.core.read_models import BenchmarkResult
from themis.core.stores.factory import create_run_store


def dump_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def load_experiment(config: str, *, overrides: list[str] | None = None) -> Experiment:
    return Experiment.from_config(config, overrides=overrides)


def initialize_store(experiment: Experiment):
    store = create_run_store(experiment.storage)
    store.initialize()
    return store


def load_benchmark_result(store, run_id: str) -> BenchmarkResult:
    projection = store.get_projection(run_id, "benchmark_result")
    if not isinstance(projection, dict):
        raise ValueError(f"Benchmark projection unavailable for run_id={run_id}")
    return BenchmarkResult.model_validate(projection)
