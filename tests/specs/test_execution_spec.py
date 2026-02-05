from __future__ import annotations

import pytest

from themis.specs import ExecutionSpec


def test_execution_spec_defaults():
    spec = ExecutionSpec()
    assert spec.workers == 4
    assert spec.max_retries == 3
    assert spec.max_in_flight_tasks is None


def test_execution_spec_workers_validation():
    with pytest.raises(ValueError, match="workers"):
        ExecutionSpec(workers=0)


def test_execution_spec_retries_validation():
    with pytest.raises(ValueError, match="max_retries"):
        ExecutionSpec(max_retries=0)


def test_execution_spec_in_flight_validation():
    with pytest.raises(ValueError, match="max_in_flight_tasks"):
        ExecutionSpec(max_in_flight_tasks=0)
