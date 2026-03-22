import pytest
from pydantic import ValidationError

import themis.specs.foundational as foundational
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    JinjaTransform,
    JudgeInferenceSpec,
    McpServerSpec,
    ModelSpec,
    RenameFieldTransform,
    TaskSpec,
    ToolSpec,
)
from themis.benchmark.query import DatasetQuerySpec
from themis.specs.experiment import InferenceParamsSpec, ItemSamplingSpec
from themis.types.enums import DatasetSource


def test_model_spec():
    spec = ModelSpec(
        model_id="gpt-4", provider="openai", extras={"api_version": "2024-02-15"}
    )
    assert spec.model_id == "gpt-4"
    assert spec.provider == "openai"
    assert spec.extras["api_version"] == "2024-02-15"


def test_tool_spec():
    spec = ToolSpec(
        id="search",
        description="Search the corpus.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        extras={"source": "project"},
    )

    assert spec.id == "search"
    assert spec.input_schema["type"] == "object"
    assert spec.extras == {"source": "project"}


def test_mcp_server_spec_supports_remote_server_and_allowed_tools():
    spec = McpServerSpec(
        id="dice",
        server_label="dice",
        server_url="https://dmcp-server.deno.dev/sse",
        allowed_tools=["roll", "stats"],
        authorization_secret_name="DICE_TOKEN",
    )

    assert spec.server_url == "https://dmcp-server.deno.dev/sse"
    assert spec.connector_id is None
    assert spec.allowed_tools == ["roll", "stats"]
    assert spec.authorization_secret_name == "DICE_TOKEN"


def test_mcp_server_spec_requires_exactly_one_connection_target():
    with pytest.raises(
        ValidationError, match="requires exactly one of server_url or connector_id"
    ):
        McpServerSpec(id="bad", server_label="bad")

    with pytest.raises(
        ValidationError, match="requires exactly one of server_url or connector_id"
    ):
        McpServerSpec(
            id="bad",
            server_label="bad",
            server_url="https://example.com/mcp",
            connector_id="connector_dropbox",
        )


def test_mcp_server_spec_normalizes_blank_connection_targets() -> None:
    by_connector = McpServerSpec(
        id="calendar",
        server_label="google_calendar",
        server_url="   ",
        connector_id="connector_googlecalendar",
    )
    assert by_connector.server_url is None
    assert by_connector.connector_id == "connector_googlecalendar"

    by_url = McpServerSpec(
        id="dice",
        server_label="dice",
        server_url="https://dmcp-server.deno.dev/sse",
        connector_id="  ",
    )
    assert by_url.server_url == "https://dmcp-server.deno.dev/sse"
    assert by_url.connector_id is None

    with pytest.raises(
        ValidationError, match="requires exactly one of server_url or connector_id"
    ):
        McpServerSpec(
            id="bad",
            server_label="bad",
            server_url=" ",
            connector_id="\t",
        )


def test_dataset_spec():
    spec = DatasetSpec(
        source="huggingface", dataset_id="gsm8k", split="test", revision="main"
    )
    assert spec.source == DatasetSource.HUGGINGFACE
    assert spec.model_dump(mode="json")["source"] == "huggingface"


def test_dataset_spec_requires_id_unless_local():
    with pytest.raises(ValidationError, match="requires a dataset_id"):
        DatasetSpec(source="huggingface", dataset_id=None)

    spec_local = DatasetSpec(source="local", dataset_id=None, data_dir="/tmp/data")
    assert spec_local.data_dir == "/tmp/data"


def test_dataset_spec_rejects_unknown_source():
    with pytest.raises(ValidationError):
        DatasetSpec(source="unsupported", dataset_id="gsm8k")


def test_task_spec():
    GenerationSpec = getattr(foundational, "GenerationSpec", None)
    assert GenerationSpec is not None

    dataset = DatasetSpec(source="local", data_dir="/tmp")
    spec = TaskSpec(
        task_id="math_eval",
        dataset=dataset,
        generation=GenerationSpec(),
        tool_ids=["search"],
        mcp_server_ids=["dice"],
    )
    assert spec.task_id == "math_eval"
    assert spec.tool_ids == ["search"]
    assert spec.mcp_server_ids == ["dice"]


def test_task_spec_validation():
    dataset = DatasetSpec(source="local", data_dir="/tmp")
    with pytest.raises(ValidationError, match="at least one stage"):
        TaskSpec(task_id="bad", dataset=dataset)


def test_task_spec_allows_transform_only():
    OutputTransformSpec = getattr(foundational, "OutputTransformSpec", None)
    assert OutputTransformSpec is not None

    task = TaskSpec(
        task_id="transform_only",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            )
        ],
    )

    assert task.generation is None
    assert len(task.output_transforms) == 1


def test_task_spec_allows_evaluation_only():
    OutputTransformSpec = getattr(foundational, "OutputTransformSpec", None)
    EvaluationSpec = getattr(foundational, "EvaluationSpec", None)
    assert OutputTransformSpec is not None
    assert EvaluationSpec is not None

    task = TaskSpec(
        task_id="eval_only",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            )
        ],
        evaluations=[EvaluationSpec(name="judge", transform="json", metrics=["em"])],
    )

    assert task.generation is None
    assert len(task.evaluations) == 1


def test_task_spec_rejects_duplicate_transform_names():
    OutputTransformSpec = getattr(foundational, "OutputTransformSpec", None)
    GenerationSpec = getattr(foundational, "GenerationSpec", None)
    assert OutputTransformSpec is not None
    assert GenerationSpec is not None

    with pytest.raises(ValidationError, match="duplicate output transform name"):
        TaskSpec(
            task_id="dup",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(extractors=["json"]),
                ),
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(extractors=["regex"]),
                ),
            ],
        )


def test_task_spec_rejects_evaluation_with_unknown_transform():
    EvaluationSpec = getattr(foundational, "EvaluationSpec", None)
    GenerationSpec = getattr(foundational, "GenerationSpec", None)
    assert EvaluationSpec is not None
    assert GenerationSpec is not None

    with pytest.raises(ValidationError, match="unknown output transform"):
        TaskSpec(
            task_id="bad_ref",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            evaluations=[
                EvaluationSpec(name="judge", transform="missing", metrics=["em"])
            ],
        )


def test_task_spec_rejects_duplicate_mcp_server_ids():
    with pytest.raises(ValidationError, match="duplicate MCP server id"):
        TaskSpec(
            task_id="dup_mcp",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=getattr(foundational, "GenerationSpec")(),
            mcp_server_ids=["dice", "dice"],
        )


def test_dataset_spec_parses_transform_hierarchy():
    spec = DatasetSpec(
        source="huggingface",
        dataset_id="gsm8k",
        transforms=[
            {
                "kind": "rename",
                "field": "rendered_question",
                "source_field": "question",
            },
            {
                "kind": "jinja",
                "field": "rendered_prompt",
                "template": "Answer: {rendered_question}",
            },
        ],
    )

    assert isinstance(spec.transforms[0], RenameFieldTransform)
    assert isinstance(spec.transforms[1], JinjaTransform)


def test_extractor_chain_spec_coerces_string_and_mapping_entries():
    chain = ExtractorChainSpec(
        extractors=[
            "regex",
            {"id": "json_schema", "config": {"schema": {"type": "object"}}},
        ]
    )

    assert chain.extractors == [
        ExtractorRefSpec(id="regex", config={}),
        ExtractorRefSpec(id="json_schema", config={"schema": {"type": "object"}}),
    ]


def test_model_spec_rejects_non_json_safe_extras():
    with pytest.raises(ValidationError):
        ModelSpec(
            model_id="gpt-4",
            provider="openai",
            extras={"bad": object()},
        )


def test_judge_inference_spec_uses_explicit_params_field():
    spec = JudgeInferenceSpec(
        model=ModelSpec(model_id="judge-model", provider="judge"),
        params={"temperature": 0.7, "max_tokens": 64},
        extras={"mode": "rubric"},
    )

    assert spec.params == InferenceParamsSpec(temperature=0.7, max_tokens=64)
    assert spec.extras == {"mode": "rubric"}


def test_task_spec_coerces_item_sampling_spec_to_dataset_query_spec() -> None:
    GenerationSpec = getattr(foundational, "GenerationSpec", None)
    assert GenerationSpec is not None

    task = TaskSpec.model_validate(
        {
            "task_id": "sampled",
            "dataset": DatasetSpec(source=DatasetSource.MEMORY),
            "dataset_query": ItemSamplingSpec.subset(count=2, seed=7),
            "generation": GenerationSpec(),
        }
    )

    assert isinstance(task.dataset_query, DatasetQuerySpec)
    assert task.dataset_query.kind.value == "subset"
    assert task.dataset_query.count == 2
    assert task.dataset_query.seed == 7


def test_task_spec_propagates_unexpected_dataset_query_errors(monkeypatch) -> None:
    GenerationSpec = getattr(foundational, "GenerationSpec", None)
    assert GenerationSpec is not None

    def _boom(cls, payload):
        del cls, payload
        raise RuntimeError("unexpected dataset query failure")

    monkeypatch.setattr(DatasetQuerySpec, "model_validate", classmethod(_boom))

    with pytest.raises(RuntimeError, match="unexpected dataset query failure"):
        TaskSpec.model_validate(
            {
                "task_id": "sampled",
                "dataset": DatasetSpec(source=DatasetSource.MEMORY),
                "dataset_query": {"kind": "all"},
                "generation": GenerationSpec(),
            }
        )
