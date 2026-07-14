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
    RunSnapshot,
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
    "RunSnapshot",
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


def test_root_exports_are_the_exact_v6_contract() -> None:
    assert set(themis.__all__) == EXPECTED_ROOT_EXPORTS
    assert all(hasattr(themis, name) for name in EXPECTED_ROOT_EXPORTS)


def test_v6_experiment_runs_with_explicit_store() -> None:
    result = _experiment().run(store=memory_store())

    assert isinstance(result, RunResult)
    assert result.status is RunStatus.COMPLETED


def test_v6_experiment_runs_with_persistent_store(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path / "runs.sqlite3")
    experiment = _experiment()
    snapshot = experiment.compile(store=store)
    result = experiment.run(store=store)

    stored = store.resume(result.run_id)
    assert stored is not None
    assert stored.snapshot == snapshot


def test_compile_defaults_to_memory_and_default_runtime_provenance() -> None:
    snapshot = _experiment().compile()

    assert snapshot.provenance.storage.target == "memory"
    assert (
        snapshot.provenance.runtime.max_concurrent_tasks == RunOptions().max_concurrency
    )


def test_compile_records_store_and_runtime_provenance(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path / "runs.sqlite3")
    options = RunOptions(max_concurrency=3, strict_determinism=True)

    snapshot = _experiment().compile(store=store, options=options)

    assert isinstance(snapshot, RunSnapshot)
    assert snapshot.provenance.storage.target == "sqlite"
    assert snapshot.provenance.storage.kwargs["path"] == str(store.path)
    assert snapshot.provenance.runtime.max_concurrent_tasks == 3
    assert snapshot.provenance.runtime.strict_determinism is True


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
        RunOptions.model_validate({field: value})


def test_package_includes_py_typed_marker() -> None:
    assert Path(themis.__file__).with_name("py.typed").is_file()
