from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import get_args, overload

import pytest

from themis import (
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    ProjectSpec,
    PromptMessage,
    SqliteBlobStorageSpec,
    ToolSpec,
    generate_config_report,
)
from themis.config_report import (
    build_config_report_document,
    config_reportable,
    render_config_report,
)
from themis.config_report.renderers import ConfigReportRenderer
from themis.config_report.types import ConfigReportFormat, ConfigReportVerbosity
from themis.specs.experiment import (
    ExperimentSpec,
    ItemSamplingSpec,
    PromptTemplateSpec,
    RuntimeContext,
)
from themis.specs.foundational import DatasetSpec, EvaluationSpec, TaskSpec
from themis.types.enums import DatasetSource, PromptRole


@pytest.fixture(autouse=True)
def _isolated_renderer_registry(monkeypatch) -> None:
    import themis.config_report.renderers as renderers_module

    monkeypatch.setattr(
        renderers_module,
        "_RENDERERS",
        renderers_module._RENDERERS.copy(),
    )


def _build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="report-demo",
        researcher_id="researcher",
        global_seed=11,
        storage=SqliteBlobStorageSpec(root_dir=".cache/report-demo"),
        execution_policy=ExecutionPolicySpec(),
    )


def _build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="arithmetic",
                dataset=DatasetSpec(
                    source=DatasetSource.MEMORY,
                    split="test",
                ),
                evaluations=[
                    EvaluationSpec(name="default", metrics=["exact_match"]),
                ],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="Answer the question.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[InferenceParamsSpec(max_tokens=32, temperature=0.2)]
        ),
        item_sampling=ItemSamplingSpec.subset(5, seed=7),
    )


def _find_child(node, name: str):
    for child in node.children:
        if child.name == name:
            return child
    raise AssertionError(f"missing child {name!r}")


@overload
def _normalized_path(path: None) -> None: ...


@overload
def _normalized_path(path: str) -> str: ...


def _normalized_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path.replace("\\", "/")


def test_normalized_path_converts_windows_separators() -> None:
    assert _normalized_path(r"D:\a\themis\themis\themis\specs\experiment.py") == (
        "D:/a/themis/themis/themis/specs/experiment.py"
    )


def test_build_config_report_document_collects_leaf_metadata() -> None:
    params = InferenceParamsSpec(max_tokens=64, temperature=0.3)

    document = build_config_report_document(params, entrypoint="tests:leaf")
    root = document.root

    assert document.header.entrypoint == "tests:leaf"
    assert document.header.root_type == "InferenceParamsSpec"
    assert root.class_name == "InferenceParamsSpec"
    assert root.source_file is not None
    normalized_root_source = _normalized_path(root.source_file)
    assert normalized_root_source is not None
    assert normalized_root_source.endswith("themis/specs/experiment.py")
    assert root.source_line == 103

    max_tokens = next(
        parameter for parameter in root.parameters if parameter.name == "max_tokens"
    )
    assert max_tokens.value == 64
    assert max_tokens.type_repr == "int"
    assert max_tokens.default == 1024
    assert max_tokens.has_default is True
    assert max_tokens.doc == "Max string length generated."
    assert max_tokens.source_file is not None
    normalized_max_tokens_source = _normalized_path(max_tokens.source_file)
    assert normalized_max_tokens_source is not None
    assert normalized_max_tokens_source.endswith("themis/specs/experiment.py")
    assert max_tokens.source_line == 113
    assert max_tokens.declared_in == "InferenceParamsSpec"


def test_build_config_report_document_collects_nested_bundle_bottom_up() -> None:
    config = {"project": _build_project(), "experiment": _build_experiment()}

    document = build_config_report_document(config, entrypoint="tests:bundle")
    root = document.root

    assert document.header.project_name == "report-demo"
    assert root.name == "config"
    assert root.path == "$"
    assert root.depth == 0

    project_node = _find_child(root, "project")
    experiment_node = _find_child(root, "experiment")
    task_node = _find_child(experiment_node, "tasks")
    task_item_node = _find_child(task_node, "[0]")

    assert project_node.parent_path == "$"
    assert project_node.depth == 1
    assert experiment_node.parent_path == "$"
    assert task_node.path == "$.experiment.tasks"
    assert task_item_node.path == "$.experiment.tasks[0]"
    assert task_item_node.depth == 3
    assert any(parameter.name == "task_id" for parameter in task_item_node.parameters)


def test_build_config_report_document_includes_declared_tools_but_hides_runtime_handlers() -> (
    None
):
    config = {
        "project": _build_project().model_copy(
            update={
                "tools": [
                    ToolSpec(
                        id="search",
                        description="Project search tool.",
                        input_schema={"type": "object"},
                    )
                ]
            }
        ),
        "runtime": RuntimeContext(
            run_labels={"phase": "smoke"},
            tool_handlers={"search": object()},
        ),
    }

    root = build_config_report_document(config, entrypoint="tests:tools").root
    project_node = _find_child(root, "project")
    runtime_node = _find_child(root, "runtime")
    tools_node = _find_child(project_node, "tools")

    assert tools_node.children
    assert all(
        parameter.name != "tool_handlers" for parameter in runtime_node.parameters
    )
    assert all(child.name != "tool_handlers" for child in runtime_node.children)


def test_build_config_report_document_resolves_inheritance_metadata() -> None:
    storage = SqliteBlobStorageSpec(root_dir=".cache/run")

    document = build_config_report_document(storage, entrypoint="tests:inheritance")
    root = document.root

    backend = next(
        parameter for parameter in root.parameters if parameter.name == "backend"
    )
    root_dir = next(
        parameter for parameter in root.parameters if parameter.name == "root_dir"
    )
    compression = next(
        parameter for parameter in root.parameters if parameter.name == "compression"
    )

    assert backend.declared_in == "SqliteBlobStorageSpec"
    assert backend.source_line is not None
    assert root_dir.declared_in == "SqliteBlobStorageSpec"
    assert root_dir.source_line is not None
    assert compression.declared_in == "_StorageSpecBase"
    assert compression.source_line is not None


def test_build_config_report_document_reads_inline_comments_from_fixture() -> None:
    from tests.config_report.fixture_module import CommentedConfig

    document = build_config_report_document(
        CommentedConfig(), entrypoint="tests:comments"
    )
    root = document.root

    alpha = next(
        parameter for parameter in root.parameters if parameter.name == "alpha"
    )
    beta = next(parameter for parameter in root.parameters if parameter.name == "beta")

    assert alpha.inline_comment == "Inline comment attached to alpha."
    assert beta.inline_comment == "Leading comment attached to beta."


def test_build_config_report_document_marks_cycles_without_recursing_forever() -> None:
    payload: dict[str, object] = {"name": "root"}
    payload["self"] = payload

    document = build_config_report_document(payload, entrypoint="tests:cycle")
    root = document.root
    cycle_node = _find_child(root, "self")

    assert cycle_node.path == "$.self"
    assert cycle_node.class_name == "cycle"
    assert any(parameter.name == "cycle_ref" for parameter in cycle_node.parameters)


def test_generate_config_report_renders_json_yaml_markdown_and_latex(
    tmp_path: Path,
) -> None:
    config = {"project": _build_project(), "experiment": _build_experiment()}

    json_output = generate_config_report(config, format="json")
    yaml_output = generate_config_report(config, format="yaml")
    markdown_output = generate_config_report(config, format="markdown")
    latex_output = generate_config_report(config, format="latex")

    payload = json.loads(json_output)
    assert payload["header"]["project_name"] == "report-demo"
    assert payload["root"]["children"][0]["name"] == "project"

    assert "header:" in yaml_output
    assert 'project_name: "report-demo"' in yaml_output
    assert "root:" in yaml_output
    assert "children:" in yaml_output

    assert "<details>" in markdown_output
    assert (
        "| Parameter | Value | Type | Default | Declared In | Source | Notes |"
        in markdown_output
    )
    assert "$.experiment" in markdown_output

    assert "\\section*{Configuration Report}" in latex_output
    assert "\\subsection*{project}" in latex_output
    assert "\\begin{longtable}" in latex_output

    output_path = tmp_path / "config-report.md"
    written = generate_config_report(config, format="markdown", output=str(output_path))
    assert output_path.read_text() == written


def test_generate_config_report_default_verbosity_keeps_paper_params_only() -> None:
    config = {"project": _build_project(), "experiment": _build_experiment()}

    json_output = generate_config_report(config, format="json", verbosity="default")
    payload = json.loads(json_output)
    header = payload["header"]
    project = payload["root"]["children"][0]
    experiment = payload["root"]["children"][1]

    assert header["verbosity"] == "default"
    assert {parameter["name"] for parameter in project["parameters"]} == {
        "project_name",
        "researcher_id",
        "global_seed",
    }
    assert "schema_version" not in {
        parameter["name"] for parameter in project["parameters"]
    }
    assert "num_samples" not in {
        parameter["name"] for parameter in experiment["parameters"]
    }
    assert "item_sampling" in {child["name"] for child in experiment["children"]}


def test_generate_config_report_full_verbosity_preserves_existing_output_shape() -> (
    None
):
    config = {"project": _build_project(), "experiment": _build_experiment()}

    default_payload = json.loads(
        generate_config_report(config, format="json", verbosity="default")
    )
    full_payload = json.loads(
        generate_config_report(config, format="json", verbosity="full")
    )
    full_project = full_payload["root"]["children"][0]

    assert full_payload["header"]["verbosity"] == "full"
    assert len(json.dumps(full_payload)) > len(json.dumps(default_payload))
    assert "schema_version" in {
        parameter["name"] for parameter in full_project["parameters"]
    }


def test_default_verbosity_prunes_empty_branches() -> None:
    config = {"project": _build_project(), "experiment": _build_experiment()}

    default_payload = json.loads(
        generate_config_report(config, format="json", verbosity="default")
    )
    full_payload = json.loads(
        generate_config_report(config, format="json", verbosity="full")
    )

    full_storage = _find_child(
        _find_child(build_config_report_document(config).root, "project"),
        "storage",
    )
    assert full_storage.parameters

    default_project = default_payload["root"]["children"][0]
    full_project = full_payload["root"]["children"][0]
    assert "storage" not in {child["name"] for child in default_project["children"]}
    assert "storage" in {child["name"] for child in full_project["children"]}


def test_render_config_report_honors_verbosity() -> None:
    document = build_config_report_document(
        {"project": _build_project(), "experiment": _build_experiment()}
    )

    default_markdown = render_config_report(
        document, format="markdown", verbosity="default"
    )
    full_markdown = render_config_report(document, format="markdown", verbosity="full")

    assert "- Verbosity: default" in default_markdown
    assert "- Verbosity: full" in full_markdown
    assert len(full_markdown) > len(default_markdown)


def test_default_verbosity_honors_explicit_paper_and_non_paper_metadata() -> None:
    from tests.config_report.fixture_module import VerbosityOverrideConfig

    default_payload = json.loads(
        generate_config_report(
            VerbosityOverrideConfig(required_param="required"),
            format="json",
            verbosity="default",
        )
    )
    root = default_payload["root"]

    assert {parameter["name"] for parameter in root["parameters"]} == {
        "forced_param",
        "required_param",
    }


def test_generate_config_report_accepts_pathlike_output(tmp_path: Path) -> None:
    config = {"project": _build_project(), "experiment": _build_experiment()}
    output_path = tmp_path / "config-report.json"

    written = generate_config_report(config, format="json", output=output_path)

    assert output_path.read_text(encoding="utf-8") == written


def test_build_config_report_document_tolerates_plain_object_without_source(
    monkeypatch,
) -> None:
    import inspect

    import themis.config_report.collector as collector_module

    class DynamicConfig:
        alpha: int

        def __init__(self) -> None:
            self.alpha = 1

    original_getsourcelines = inspect.getsourcelines

    def raising_getsourcelines(obj):
        if obj is DynamicConfig:
            raise OSError("source unavailable")
        return original_getsourcelines(obj)

    monkeypatch.setattr(collector_module.inspect, "getsourcefile", lambda obj: None)
    monkeypatch.setattr(
        collector_module.inspect, "getsourcelines", raising_getsourcelines
    )

    document = build_config_report_document(
        DynamicConfig(), entrypoint="tests:dynamic-object"
    )
    root = document.root

    assert root.class_name == "DynamicConfig"
    assert root.source_file is None
    assert root.source_line is None
    assert (
        next(
            parameter for parameter in root.parameters if parameter.name == "alpha"
        ).value
        == 1
    )


def test_build_config_report_document_tolerates_dataclass_without_source(
    monkeypatch,
) -> None:
    import inspect

    import themis.config_report.collector as collector_module

    @dataclasses.dataclass(frozen=True)
    class DynamicDataclass:
        alpha: int = 1

    original_getsourcelines = inspect.getsourcelines

    def raising_getsourcelines(obj):
        if obj is DynamicDataclass:
            raise OSError("source unavailable")
        return original_getsourcelines(obj)

    monkeypatch.setattr(collector_module.inspect, "getsourcefile", lambda obj: None)
    monkeypatch.setattr(
        collector_module.inspect, "getsourcelines", raising_getsourcelines
    )

    document = build_config_report_document(
        DynamicDataclass(), entrypoint="tests:dynamic-dataclass"
    )
    root = document.root

    assert root.class_name == "DynamicDataclass"
    assert root.source_file is None
    assert root.source_line is None


def test_build_config_report_document_tolerates_pydantic_model_without_source(
    monkeypatch,
) -> None:
    import inspect

    from pydantic import BaseModel
    import themis.config_report.collector as collector_module

    class DynamicModel(BaseModel):
        alpha: int = 1

    original_getsourcelines = inspect.getsourcelines

    def raising_getsourcelines(obj):
        if obj is DynamicModel:
            raise OSError("source unavailable")
        return original_getsourcelines(obj)

    monkeypatch.setattr(collector_module.inspect, "getsourcefile", lambda obj: None)
    monkeypatch.setattr(
        collector_module.inspect, "getsourcelines", raising_getsourcelines
    )

    document = build_config_report_document(
        DynamicModel(), entrypoint="tests:dynamic-pydantic"
    )
    root = document.root

    assert root.class_name == "DynamicModel"
    assert root.source_file is None
    assert root.source_line is None


def test_build_config_report_document_tolerates_sourcefile_type_error(
    monkeypatch,
) -> None:
    import themis.config_report.collector as collector_module

    @dataclasses.dataclass(frozen=True)
    class DynamicDataclass:
        alpha: int = 1

    original_getsourcefile = collector_module.inspect.getsourcefile

    def raising_getsourcefile(obj):
        if obj is DynamicDataclass:
            raise TypeError("built-in class")
        return original_getsourcefile(obj)

    monkeypatch.setattr(
        collector_module.inspect,
        "getsourcefile",
        raising_getsourcefile,
    )

    root = build_config_report_document(DynamicDataclass()).root

    assert root.class_name == "DynamicDataclass"
    assert root.source_file is None


def test_build_config_report_document_tolerates_source_index_type_error(
    monkeypatch,
) -> None:
    import themis.config_report.collector as collector_module

    @dataclasses.dataclass(frozen=True)
    class DynamicDataclass:
        alpha: int = 1

    class BrokenSourceIndex:
        def get_class_info(self, qualname: str):
            raise TypeError(f"bad source index for {qualname}")

    monkeypatch.setattr(
        collector_module,
        "load_source_index",
        lambda path: BrokenSourceIndex(),
    )

    root = build_config_report_document(DynamicDataclass()).root

    assert root.class_name == "DynamicDataclass"
    assert root.source_file is not None
    assert root.source_line is not None


def test_build_config_report_document_redacts_defaults_and_nested_children() -> None:
    @dataclasses.dataclass(frozen=True)
    class SecretChild:
        token: str = "nested-secret"

    @config_reportable(redacted_fields={"secret_value", "secret_child"})
    @dataclasses.dataclass(frozen=True)
    class SecretConfig:
        secret_value: str = "top-secret"
        secret_child: SecretChild = dataclasses.field(default_factory=SecretChild)

    root = build_config_report_document(SecretConfig(secret_child=SecretChild())).root

    secret_value = next(
        parameter for parameter in root.parameters if parameter.name == "secret_value"
    )
    secret_child = _find_child(root, "secret_child")
    child_token = next(
        parameter for parameter in secret_child.parameters if parameter.name == "token"
    )

    assert secret_value.value == "***REDACTED***"
    assert secret_value.default == "***REDACTED***"
    assert child_token.value == "***REDACTED***"


def test_build_config_report_document_redacts_secret_like_mapping_keys() -> None:
    root = build_config_report_document(
        {
            "api_key": "top-secret",
            "nested": {
                "token": "nested-secret",
            },
        }
    ).root

    api_key = next(
        parameter for parameter in root.parameters if parameter.name == "api_key"
    )
    nested = _find_child(root, "nested")
    token = next(
        parameter for parameter in nested.parameters if parameter.name == "token"
    )

    assert api_key.value == "***REDACTED***"
    assert token.value == "***REDACTED***"


def test_build_config_report_document_reports_default_factory_without_calling_it() -> (
    None
):
    factory_calls = 0

    def noisy_factory() -> str:
        nonlocal factory_calls
        factory_calls += 1
        return "generated"

    @dataclasses.dataclass(frozen=True)
    class FactoryConfig:
        value: str = dataclasses.field(default_factory=noisy_factory)

    root = build_config_report_document(FactoryConfig(value="explicit")).root
    value = next(
        parameter for parameter in root.parameters if parameter.name == "value"
    )

    assert factory_calls == 0
    assert value.has_default is True
    assert value.default == "<factory: noisy_factory>"


def test_default_verbosity_uses_mro_precedence_for_visibility_overrides() -> None:
    @config_reportable(paper_fields={"foo"})
    @dataclasses.dataclass(frozen=True)
    class BaseVisibilityConfig:
        foo: int = 1

    @config_reportable(non_paper_fields={"foo"})
    @dataclasses.dataclass(frozen=True)
    class DerivedVisibilityConfig(BaseVisibilityConfig):
        pass

    payload = json.loads(
        generate_config_report(
            DerivedVisibilityConfig(),
            format="json",
            verbosity="default",
        )
    )

    assert payload["root"]["parameters"] == []


def test_default_verbosity_does_not_hide_ordinary_param_by_name() -> None:
    @dataclasses.dataclass(frozen=True)
    class OrdinaryParamConfig:
        ordinary_param: int = 1

    payload = json.loads(
        generate_config_report(
            OrdinaryParamConfig(ordinary_param=2),
            format="json",
            verbosity="default",
        )
    )

    assert {parameter["name"] for parameter in payload["root"]["parameters"]} == {
        "ordinary_param"
    }


def test_latex_renderer_escapes_special_tokens_without_double_escaping() -> None:
    latex_output = generate_config_report(
        {"project": {"path": r"C:\tmp\demo", "marker": "~^"}},
        format="latex",
    )

    assert r"\textbackslash{}" in latex_output
    assert r"\textasciitilde{}" in latex_output
    assert r"\textasciicircum{}" in latex_output
    assert r"\textbackslash\{\}" not in latex_output


def test_load_source_index_respects_declared_file_encoding(tmp_path: Path) -> None:
    from themis.config_report.source_index import load_source_index

    source_path = tmp_path / "latin1_module.py"
    source_path.write_bytes(
        (
            b"# -*- coding: latin-1 -*-\n"
            b"class EncodedConfig:\n"
            b"    field: str = 'caf\xe9'\n"
        )
    )

    class_info = load_source_index(str(source_path)).get_class_info("EncodedConfig")

    assert class_info is not None
    assert class_info.field_lines["field"] == 3


class _XmlRenderer:
    def render(self, document) -> str:
        return f"<report root='{document.root.name}' />\n"


def test_render_config_report_supports_registered_custom_renderer() -> None:
    from themis.config_report import (
        get_config_report_renderer,
        register_config_report_renderer,
        render_config_report,
    )

    register_config_report_renderer("xml", _XmlRenderer())
    document = build_config_report_document({"project": _build_project()})

    assert isinstance(get_config_report_renderer("xml"), _XmlRenderer)
    assert render_config_report(document, format="xml") == "<report root='config' />\n"


def test_register_config_report_renderer_rejects_duplicate_name() -> None:
    import pytest

    from themis.config_report import register_config_report_renderer

    register_config_report_renderer("xml-dupe", _XmlRenderer())

    with pytest.raises(ValueError, match="already registered"):
        register_config_report_renderer("xml-dupe", _XmlRenderer())


def test_register_config_report_renderer_can_overwrite_existing_name() -> None:
    from themis.config_report import (
        get_config_report_renderer,
        register_config_report_renderer,
    )

    class ReplacementRenderer:
        def render(self, document) -> str:
            return f"<replacement root='{document.root.name}' />\n"

    register_config_report_renderer("xml-overwrite", _XmlRenderer())
    register_config_report_renderer(
        "xml-overwrite", ReplacementRenderer(), overwrite=True
    )

    assert isinstance(get_config_report_renderer("xml-overwrite"), ReplacementRenderer)


def test_list_config_report_renderers_includes_builtins_and_custom_registration() -> (
    None
):
    from themis.config_report import (
        list_config_report_renderers,
        register_config_report_renderer,
    )

    register_config_report_renderer("xml-list", _XmlRenderer())

    assert {"json", "yaml", "markdown", "latex", "xml-list"} <= set(
        list_config_report_renderers()
    )


def test_get_config_report_renderer_lists_live_registry_formats_in_error() -> None:
    import pytest

    from themis.config_report import register_config_report_renderer
    from themis.config_report.renderers import get_config_report_renderer

    register_config_report_renderer("xml-live", _XmlRenderer())

    with pytest.raises(ValueError, match="json, latex, markdown, xml-live, yaml"):
        get_config_report_renderer("missing")


def test_config_report_renderer_protocol_is_publicly_usable() -> None:
    renderer: ConfigReportRenderer = _XmlRenderer()

    assert renderer.render(
        build_config_report_document({"project": _build_project()})
    ) == ("<report root='config' />\n")


def test_config_report_public_type_aliases_are_stable() -> None:
    assert get_args(ConfigReportFormat) == ("json", "yaml", "markdown", "latex")
    assert get_args(ConfigReportVerbosity) == ("default", "full")
