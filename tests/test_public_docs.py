from __future__ import annotations

import inspect
from collections.abc import Callable

from themis import __all__ as root_all
from themis import __version__
from themis.catalog import (
    builtin_component_refs,
    get_benchmark,
    list_benchmark_ids,
    list_benchmarks,
    list_component_ids,
    load,
    run,
)
from themis.cli import main
from themis.cli.helpers import (
    dump_json,
    initialize_store,
    load_benchmark_result,
    load_experiment,
)
from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
    SelectContext,
)
from themis.core.events import RunEvent, RunStartedEvent, event_from_dict
from themis.core.protocols import (
    CandidateReducer,
    CandidateSelector,
    EvaluationWorkflow,
    Generator,
    JudgeModel,
    LLMMetric,
    Parser,
    PureMetric,
    TraceMetric,
    TracingProvider,
    WorkflowRunner,
)
from themis.core.reporter import Reporter, snapshot_report
from themis.core.stores.base import ProjectionRefreshingStore
from themis.core.stores.factory import (
    available_store_backends,
    create_run_store,
    memory_store,
    register_store_backend,
)
from themis.core.stores.jsonl import JsonlRunStore
from themis.core.stores.mongodb import MongoDbRunStore
from themis.core.stores.postgres import PostgresRunStore


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
    assert __version__ == "4.0.0"


def test_catalog_entrypoints_are_documented_and_typed() -> None:
    for symbol in (
        load,
        run,
        builtin_component_refs,
        list_component_ids,
        list_benchmark_ids,
        list_benchmarks,
        get_benchmark,
    ):
        _assert_docstring(symbol)
        _assert_annotations(symbol)


def test_reporting_surface_is_documented_and_typed() -> None:
    _assert_docstring(snapshot_report)
    _assert_annotations(snapshot_report)
    _assert_docstring(Reporter)
    for method_name in (
        "export_json",
        "export_markdown",
        "export_csv",
        "export_latex",
        "export_score_table",
    ):
        _assert_docstring(getattr(Reporter, method_name))


def test_context_models_have_docstrings() -> None:
    for context_model in (
        GenerateContext,
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
        LLMMetric,
        TraceMetric,
        WorkflowRunner,
        TracingProvider,
    ):
        _assert_docstring(protocol)


def test_events_and_store_surface_have_docstrings() -> None:
    for symbol in (
        RunEvent,
        RunStartedEvent,
        event_from_dict,
        ProjectionRefreshingStore,
        memory_store,
        register_store_backend,
        available_store_backends,
        create_run_store,
        JsonlRunStore,
        MongoDbRunStore,
        PostgresRunStore,
        dump_json,
        load_experiment,
        initialize_store,
        load_benchmark_result,
    ):
        _assert_docstring(symbol)


def test_cli_entrypoint_has_docstring_and_annotation() -> None:
    _assert_docstring(main)
    _assert_annotations(main)
