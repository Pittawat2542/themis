from __future__ import annotations

from themis.catalog import run
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore


def test_catalog_run_executes_manifest_benchmark_end_to_end() -> None:
    store = InMemoryRunStore()

    result = run("mmlu_pro", store=store)

    assert result.status is RunStatus.COMPLETED
    assert result.progress.total_cases == 1
