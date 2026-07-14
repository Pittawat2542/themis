from __future__ import annotations

import inspect
from collections.abc import Callable

from themis import __all__ as root_all
from themis import __version__
from themis.analysis import (
    Reporter,
    ReporterProtocol,
    available_reporters,
    create_reporter,
    get_attempt_history,
    get_case_audit,
    get_evaluation_execution,
    get_execution_state,
    get_projection,
    get_run_record,
    get_run_snapshot,
    get_score_claim_history,
    get_telemetry_summary,
    query_run_records,
    register_reporter,
    resolve_run_id,
    resolve_run_record,
    snapshot_report,
)
from themis.catalog import (
    builtin_component_refs,
    get_benchmark,
    list_benchmark_ids,
    list_benchmarks,
    list_component_ids,
    load,
    run,
    validate_benchmark,
)
from themis.cli import main
from themis.components import (
    CandidateReducer,
    CandidateSelector,
    EvalScoreContext,
    EvaluationWorkflow,
    GenerationContext,
    Generator,
    JudgeModel,
    ParseContext,
    Parser,
    PureMetric,
    ReduceContext,
    ScoreContext,
    SelectContext,
    WorkflowMetric,
    WorkflowRunner,
)
from themis.runtime import TracingProvider
from themis.storage import (
    JsonlRunStore,
    MongoDbRunStore,
    PostgresRunStore,
    RunStoreBase,
    available_store_backends,
    create_run_store,
    memory_store,
    register_store_backend,
)
from tests.release import CURRENT_VERSION


def _assert_docstring(value: object) -> None:
    assert inspect.getdoc(value), f"missing docstring for {value!r}"


def _assert_annotations(value: Callable[..., object]) -> None:
    signature = inspect.signature(value)
    for parameter in signature.parameters.values():
        assert parameter.annotation is not inspect.Signature.empty, (
            f"missing annotation for {value!r} parameter {parameter.name}"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"missing return annotation for {value!r}"
    )


def test_root_package_exposes_version() -> None:
    assert "__version__" in root_all
    assert __version__ == CURRENT_VERSION


def test_catalog_entrypoints_are_documented_and_typed() -> None:
    for symbol in (
        load,
        run,
        builtin_component_refs,
        list_component_ids,
        list_benchmark_ids,
        list_benchmarks,
        get_benchmark,
        validate_benchmark,
    ):
        _assert_docstring(symbol)
        _assert_annotations(symbol)


def test_reporting_surface_is_documented_and_typed() -> None:
    _assert_docstring(snapshot_report)
    _assert_annotations(snapshot_report)
    _assert_docstring(Reporter)
    _assert_docstring(ReporterProtocol)
    for symbol in (register_reporter, available_reporters, create_reporter):
        _assert_docstring(symbol)
        _assert_annotations(symbol)
    for method_name in (
        "summary",
        "score_rows",
        "export_json",
        "export_markdown",
        "export_csv",
        "export_latex",
    ):
        _assert_docstring(getattr(Reporter, method_name))


def test_context_models_have_docstrings() -> None:
    for context_model in (
        GenerationContext,
        SelectContext,
        ReduceContext,
        ParseContext,
        ScoreContext,
        EvalScoreContext,
    ):
        _assert_docstring(context_model)


def test_extension_protocols_have_docstrings() -> None:
    for protocol in (
        Generator,
        Parser,
        CandidateReducer,
        CandidateSelector,
        EvaluationWorkflow,
        JudgeModel,
        PureMetric,
        WorkflowMetric,
        WorkflowRunner,
        TracingProvider,
    ):
        _assert_docstring(protocol)


def test_store_surface_has_docstrings() -> None:
    for symbol in (
        RunStoreBase,
        memory_store,
        register_store_backend,
        available_store_backends,
        create_run_store,
        JsonlRunStore,
        MongoDbRunStore,
        PostgresRunStore,
    ):
        _assert_docstring(symbol)


def test_inspection_surface_has_docstrings() -> None:
    for symbol in (
        get_attempt_history,
        get_case_audit,
        get_evaluation_execution,
        get_execution_state,
        get_projection,
        get_run_record,
        get_run_snapshot,
        get_score_claim_history,
        get_telemetry_summary,
        query_run_records,
        resolve_run_id,
        resolve_run_record,
    ):
        _assert_docstring(symbol)


def test_cli_entrypoint_has_docstring_and_annotation() -> None:
    _assert_docstring(main)
    _assert_annotations(main)
