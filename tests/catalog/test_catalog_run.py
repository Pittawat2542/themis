from __future__ import annotations

import pytest

from themis.catalog import run
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore
from tests.catalog_ids import catalog_benchmark_ids


@pytest.mark.parametrize("benchmark_id", catalog_benchmark_ids())
def test_catalog_run_executes_manifest_benchmark_end_to_end(
    benchmark_id: str,
) -> None:
    store = InMemoryRunStore()

    result = run(benchmark_id, store=store)

    assert result.status is RunStatus.COMPLETED
    assert result.progress.total_cases == 1
