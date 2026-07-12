from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import themis
from themis import (
    Case,
    Dataset,
    Evaluation,
    Experiment,
    Generation,
    RunOptions,
    RunResult,
    evaluate,
)
from themis.core.results import RunStatus
from themis.storage import memory_store, sqlite_store


EXPECTED_ROOT_EXPORTS = {
    "Case",
    "Dataset",
    "DatasetSource",
    "Evaluation",
    "Experiment",
    "Generation",
    "MetricResult",
    "MetricInterpretation",
    "RunOptions",
    "RunResult",
    "__version__",
    "evaluate",
}


def _experiment() -> Experiment:
    return Experiment(
        datasets=[
            Dataset(
                dataset_id="sample",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"answer": "4"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        generation=Generation(generator="builtin/demo_generator"),
        evaluation=Evaluation(
            parser="builtin/json_identity",
            metrics=["builtin/exact_match"],
        ),
    )


def test_root_exports_are_the_exact_v5_contract() -> None:
    assert set(themis.__all__) == EXPECTED_ROOT_EXPORTS
    assert all(hasattr(themis, name) for name in EXPECTED_ROOT_EXPORTS)


def test_v5_experiment_runs_with_explicit_store() -> None:
    result = _experiment().run(store=memory_store())

    assert isinstance(result, RunResult)
    assert result.status is RunStatus.COMPLETED


def test_v5_experiment_runs_with_persistent_store(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path / "runs.sqlite3")
    result = _experiment().run(store=store)

    assert store.resume(result.run_id) is not None


def test_evaluate_is_deliberately_small() -> None:
    result = evaluate(
        cases=[Case(case_id="case-1", input="ok", expected_output="ok")],
        generator="builtin/demo_generator",
        metrics=["builtin/exact_match"],
        parser="builtin/json_identity",
    )

    assert result.status is RunStatus.COMPLETED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_concurrency", 0),
        ("provider_timeout_seconds", 0),
        ("generation_retry_attempts", 0),
        ("generation_retry_delay", -1),
        ("generation_retry_backoff", 0.5),
    ],
)
def test_run_options_reject_invalid_limits(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RunOptions(**{field: value})


def test_package_includes_py_typed_marker() -> None:
    assert Path(themis.__file__).with_name("py.typed").is_file()
