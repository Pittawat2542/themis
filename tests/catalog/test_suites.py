from __future__ import annotations

import pytest

from themis.catalog import (
    SuiteDefinition,
    SuiteItem,
    expand_suite,
    get_suite,
    list_suites,
    register_suite,
    run_suite,
)
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore


def test_builtin_suites_expand_to_benchmark_items() -> None:
    suite_ids = list_suites()

    assert "math-core" in suite_ids
    assert "qa-core" in suite_ids

    expansion = expand_suite("math-core")

    assert expansion.suite_id == "math-core"
    assert expansion.benchmark_ids
    assert all(item.source_suite_ids == ["math-core"] for item in expansion.items)


def test_custom_suite_expansion_is_deterministic_and_nested() -> None:
    register_suite(
        SuiteDefinition(
            suite_id="test-suite-inner",
            items=[SuiteItem(benchmark_id="mmlu_pro")],
            tags=["test"],
        )
    )
    register_suite(
        SuiteDefinition(
            suite_id="test-suite-outer",
            items=[
                SuiteItem(suite_id="test-suite-inner"),
                SuiteItem(benchmark_id="frontierscience"),
            ],
            tags=["test"],
        )
    )

    first = expand_suite("test-suite-outer")
    second = expand_suite("test-suite-outer")

    assert [item.benchmark_id for item in first.items] == [
        "mmlu_pro",
        "frontierscience",
    ]
    assert first == second
    assert get_suite("test-suite-inner").suite_id == "test-suite-inner"


def test_suite_registration_rejects_cycles_and_benchmark_collisions() -> None:
    with pytest.raises(ValueError, match="collides with a benchmark"):
        register_suite(SuiteDefinition(suite_id="mmlu_pro", items=[]))

    register_suite(
        SuiteDefinition(
            suite_id="test-suite-cycle-a",
            items=[SuiteItem(suite_id="test-suite-cycle-b")],
            tags=["test"],
        )
    )
    register_suite(
        SuiteDefinition(
            suite_id="test-suite-cycle-b",
            items=[SuiteItem(suite_id="test-suite-cycle-a")],
            tags=["test"],
        )
    )

    with pytest.raises(ValueError, match="cycle"):
        expand_suite("test-suite-cycle-a")


def test_run_suite_executes_each_benchmark_as_normal_run() -> None:
    register_suite(
        SuiteDefinition(
            suite_id="test-suite-run",
            items=[SuiteItem(benchmark_id="mmlu_pro"), SuiteItem(benchmark_id="babe")],
            tags=["test"],
        )
    )
    store = InMemoryRunStore()

    result = run_suite("test-suite-run", store=store)

    assert result.suite_id == "test-suite-run"
    assert len(result.run_ids) == 2
    assert all(item.status is RunStatus.COMPLETED for item in result.runs)
    for run_id in result.run_ids:
        record = store.get_run_record(run_id)
        assert record is not None
        assert "suite:test-suite-run" in record.tags
