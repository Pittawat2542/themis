from __future__ import annotations

import pytest

from themis.specs import ExecutionSpec


def test_execution_spec_defaults():
    spec = ExecutionSpec()
    assert spec.workers == 4
    assert spec.max_retries == 3


def test_execution_spec_workers_validation():
    with pytest.raises(ValueError, match="workers"):
        ExecutionSpec(workers=0)


def test_execution_spec_retries_validation():
    with pytest.raises(ValueError, match="max_retries"):
        ExecutionSpec(max_retries=0)
