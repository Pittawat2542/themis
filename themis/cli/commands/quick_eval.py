"""Quick-eval CLI commands."""

from __future__ import annotations

import json
from typing import cast

from cyclopts import App

from themis import InMemoryRunStore
from themis.catalog import run as run_catalog_benchmark
from themis.core.base import JSONValue
from themis.core.config import StorageConfig
from themis.core.dataset_inputs import (
    MissingOptionalDependencyError,
    dataset_from_huggingface,
    dataset_from_inline,
    dataset_from_jsonl,
)
from themis.core.evaluate import evaluate

quick_eval_app = App(name="quick-eval", help="Quick evaluation workflows.")
_DEFAULT_STORAGE = StorageConfig(store="memory")


def _result_payload(*, run_id: str, status: str, metric_means: dict[str, float]) -> str:
    return json.dumps(
        {
            "run_id": run_id,
            "status": status,
            "metric_means": metric_means,
        },
        sort_keys=True,
    )


def _run_dataset(dataset) -> str:
    store = InMemoryRunStore()
    result = evaluate(
        model="builtin/demo_generator",
        data=[dataset],
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=_DEFAULT_STORAGE,
        store=store,
    )
    benchmark = store.get_projection(result.run_id, "benchmark_result")
    metric_means = _metric_means_from_projection(benchmark)
    return _result_payload(
        run_id=result.run_id, status=result.status.value, metric_means=metric_means
    )


@quick_eval_app.command
def inline(*, input_json: str, expected_output_json: str | None = None) -> int:
    dataset = dataset_from_inline(
        input_value=json.loads(input_json),
        expected_output=None
        if expected_output_json is None
        else json.loads(expected_output_json),
    )
    print(_run_dataset(dataset))
    return 0


@quick_eval_app.command
def file(*, path: str) -> int:
    dataset = dataset_from_jsonl(path)
    print(_run_dataset(dataset))
    return 0


@quick_eval_app.command
def huggingface(
    *,
    dataset: str,
    split: str,
    input_field: str,
    expected_output_field: str | None = None,
    case_id_field: str | None = None,
) -> int:
    try:
        loaded_dataset = dataset_from_huggingface(
            dataset_name=dataset,
            split=split,
            input_field=input_field,
            expected_output_field=expected_output_field,
            case_id_field=case_id_field,
        )
    except MissingOptionalDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    print(_run_dataset(loaded_dataset))
    return 0


@quick_eval_app.command
def benchmark(*, name: str) -> int:
    store = InMemoryRunStore()
    result = run_catalog_benchmark(name, store=store)
    benchmark_result = store.get_projection(result.run_id, "benchmark_result")
    metric_means = _metric_means_from_projection(benchmark_result)
    print(
        _result_payload(
            run_id=result.run_id, status=result.status.value, metric_means=metric_means
        )
    )
    return 0


def _metric_means_from_projection(projection: JSONValue | None) -> dict[str, float]:
    if not isinstance(projection, dict):
        return {}
    metric_means = projection.get("metric_means")
    if not isinstance(metric_means, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in metric_means.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            cleaned[key] = float(value)
    return cast(dict[str, float], cleaned)
