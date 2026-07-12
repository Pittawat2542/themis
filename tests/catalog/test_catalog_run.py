from __future__ import annotations

import pytest

from themis.catalog import load, run
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore
from tests.catalog_ids import catalog_benchmark_ids


@pytest.mark.slow
@pytest.mark.parametrize("benchmark_id", catalog_benchmark_ids())
def test_catalog_run_executes_manifest_benchmark_end_to_end(
    benchmark_id: str,
) -> None:
    store = InMemoryRunStore()

    result = run(benchmark_id, store=store)

    definition = load(benchmark_id)
    expected_status = (
        RunStatus.PARTIAL_FAILURE
        if definition.requires_code_execution
        else RunStatus.COMPLETED
    )
    assert result.status is expected_status
    assert result.progress.total_cases >= 1
