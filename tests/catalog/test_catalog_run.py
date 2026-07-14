from __future__ import annotations

import pytest

from themis.catalog import get_benchmark, run
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore
from tests.catalog_ids import catalog_benchmark_ids


@pytest.mark.slow
@pytest.mark.parametrize("benchmark_id", catalog_benchmark_ids())
def test_catalog_run_executes_manifest_benchmark_end_to_end(
    benchmark_id: str,
    catalog_fixture_loader: None,
) -> None:
    store = InMemoryRunStore()

    result = run(benchmark_id, store=store)

    definition = get_benchmark(benchmark_id)
    expected_status = (
        RunStatus.PARTIAL_FAILURE
        if definition.requires_code_execution
        else RunStatus.COMPLETED
    )
    stored = store.resume(result.run_id)

    assert result.status is expected_status
    assert result.progress.total_cases >= 1
    assert result.cases
    assert all(case.generated_candidates for case in result.cases)
    assert stored is not None
    assert stored.snapshot.datasets
    assert stored.snapshot.datasets[0].metadata["benchmark_id"] == benchmark_id
    if definition.requires_code_execution:
        assert any(
            "explicit sandbox executor" in str(score)
            for case in result.cases
            for score in case.metric_results
        )
