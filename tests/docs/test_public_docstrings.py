from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCSTRING_TARGETS = [
    PROJECT_ROOT / "themis/errors/exceptions.py",
    PROJECT_ROOT / "themis/errors/mapping.py",
    PROJECT_ROOT / "themis/contracts/protocols.py",
    PROJECT_ROOT / "themis/orchestration/orchestrator.py",
    PROJECT_ROOT / "themis/orchestration/executor.py",
    PROJECT_ROOT / "themis/orchestration/trial_runner.py",
    PROJECT_ROOT / "themis/orchestration/trial_planner.py",
    PROJECT_ROOT / "themis/orchestration/projection_handler.py",
    PROJECT_ROOT / "themis/orchestration/candidate_pipeline.py",
    PROJECT_ROOT / "themis/orchestration/run_manifest.py",
    PROJECT_ROOT / "themis/orchestration/run_services.py",
    PROJECT_ROOT / "themis/runtime/experiment_result.py",
    PROJECT_ROOT / "themis/runtime/comparison.py",
    PROJECT_ROOT / "themis/runtime/result_services.py",
    PROJECT_ROOT / "themis/registry/plugin_registry.py",
    PROJECT_ROOT / "themis/registry/compatibility.py",
    PROJECT_ROOT / "themis/progress/bus.py",
    PROJECT_ROOT / "themis/progress/models.py",
    PROJECT_ROOT / "themis/progress/tracker.py",
    PROJECT_ROOT / "themis/storage/event_repo.py",
    PROJECT_ROOT / "themis/storage/projection_repo.py",
    PROJECT_ROOT / "themis/storage/artifact_store.py",
    PROJECT_ROOT / "themis/storage/sqlite_schema.py",
    PROJECT_ROOT / "themis/storage/postgres/manager.py",
    PROJECT_ROOT / "themis/report/builder.py",
    PROJECT_ROOT / "themis/report/exporters.py",
    PROJECT_ROOT / "themis/config_report/api.py",
    PROJECT_ROOT / "themis/config_report/collector.py",
    PROJECT_ROOT / "themis/config_report/renderers.py",
    PROJECT_ROOT / "themis/config_report/types.py",
    PROJECT_ROOT / "themis/cli/quickcheck.py",
    PROJECT_ROOT / "themis/cli/report.py",
    PROJECT_ROOT / "themis/cli/main.py",
    PROJECT_ROOT / "themis/extractors/builtin.py",
    PROJECT_ROOT / "themis/records/conversation.py",
    PROJECT_ROOT / "themis/stats/stats_engine.py",
    PROJECT_ROOT / "themis/specs/experiment.py",
    PROJECT_ROOT / "themis/specs/foundational.py",
    PROJECT_ROOT / "themis/types/enums.py",
    PROJECT_ROOT / "themis/types/events.py",
]

QUALITY_EXPECTATIONS = {
    PROJECT_ROOT / "themis/cli/main.py": {
        "build_parser",
        "main",
    },
    PROJECT_ROOT / "themis/cli/quickcheck.py": {
        "add_quickcheck_arguments",
        "configure_quickcheck_parser",
        "build_parser",
        "add_quickcheck_subparser",
        "run_with_args",
        "main",
    },
    PROJECT_ROOT / "themis/cli/report.py": {
        "add_report_arguments",
        "configure_report_parser",
        "build_parser",
        "add_report_subparser",
        "run_with_args",
        "main",
    },
    PROJECT_ROOT / "themis/config_report/api.py": {
        "render_config_report",
        "generate_config_report",
    },
    PROJECT_ROOT / "themis/runtime/experiment_result.py": {
        "ExperimentResult.compare",
        "ExperimentResult.for_transform",
        "ExperimentResult.for_evaluation",
        "ExperimentResult.export_json",
        "ExperimentResult.report",
    },
    PROJECT_ROOT / "themis/telemetry/bus.py": {
        "TelemetryBus.subscribe",
        "TelemetryBus.unsubscribe",
        "TelemetryBus.emit",
    },
    PROJECT_ROOT / "themis/telemetry/langfuse_callback.py": {
        "LangfuseCallback.subscribe",
        "LangfuseCallback.on_event",
    },
}


def _annotation_is_explicit_none(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant):
        return annotation.value is None
    if isinstance(annotation, ast.Name):
        return annotation.id == "None"
    return False


def _iter_relevant_body_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    relevant_nodes: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        relevant_nodes.append(child)
        stack.extend(ast.iter_child_nodes(child))
    return relevant_nodes


def _requires_returns_section(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is not None and not _annotation_is_explicit_none(node.returns):
        return True
    return any(
        isinstance(child, ast.Return) and child.value is not None
        for child in _iter_relevant_body_nodes(node)
    )


def _requires_raises_section(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(child, ast.Raise) for child in _iter_relevant_body_nodes(node)
    )


def test_public_reference_modules_have_public_docstrings() -> None:
    missing: list[str] = []

    for path in DOCSTRING_TARGETS:
        tree = ast.parse(path.read_text())
        if ast.get_docstring(tree) is None:
            missing.append(f"{path.relative_to(PROJECT_ROOT)}:module")

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(PROJECT_ROOT)}:{node.name}")
                for child in node.body:
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and not child.name.startswith("_"):
                        if ast.get_docstring(child) is None:
                            missing.append(
                                f"{path.relative_to(PROJECT_ROOT)}:{node.name}.{child.name}"
                            )
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(PROJECT_ROOT)}:{node.name}")

    assert missing == []


def test_key_reference_docstrings_include_google_style_sections() -> None:
    missing_sections: list[str] = []

    for path, expected_symbols in QUALITY_EXPECTATIONS.items():
        tree = ast.parse(path.read_text())
        seen: set[str] = set()

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = node.name
                if symbol in expected_symbols:
                    seen.add(symbol)
                    doc = ast.get_docstring(node) or ""
                    has_user_args = len(node.args.args) + len(node.args.kwonlyargs) > 0
                    if has_user_args and "Args:" not in doc:
                        missing_sections.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Args"
                        )
                    if _requires_returns_section(node) and "Returns:" not in doc:
                        missing_sections.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Returns"
                        )
                    if _requires_raises_section(node) and "Raises:" not in doc:
                        missing_sections.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Raises"
                        )
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbol = f"{node.name}.{child.name}"
                        if symbol in expected_symbols:
                            seen.add(symbol)
                            doc = ast.get_docstring(child) or ""
                            positional_args = (
                                child.args.args[1:] if child.args.args else []
                            )
                            has_user_args = (
                                len(positional_args) + len(child.args.kwonlyargs) > 0
                            )
                            if has_user_args and "Args:" not in doc:
                                missing_sections.append(
                                    f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Args"
                                )
                            if (
                                _requires_returns_section(child)
                                and "Returns:" not in doc
                            ):
                                missing_sections.append(
                                    f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Returns"
                                )
                            if _requires_raises_section(child) and "Raises:" not in doc:
                                missing_sections.append(
                                    f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Raises"
                                )

        missing_symbols = sorted(expected_symbols - seen)
        missing_sections.extend(
            f"{path.relative_to(PROJECT_ROOT)}:{symbol}:Missing"
            for symbol in missing_symbols
        )

    assert missing_sections == []
