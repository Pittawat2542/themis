import pytest
from pydantic import ValidationError

from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    JinjaTransform,
    JudgeInferenceSpec,
    ModelSpec,
    RenameFieldTransform,
    TaskSpec,
)
from themis.specs.experiment import InferenceParamsSpec


def test_model_spec():
    spec = ModelSpec(
        model_id="gpt-4", provider="openai", extras={"api_version": "2024-02-15"}
    )
    assert spec.model_id == "gpt-4"
    assert spec.provider == "openai"
    assert spec.extras["api_version"] == "2024-02-15"


def test_dataset_spec():
    spec = DatasetSpec(
        source="huggingface", dataset_id="gsm8k", split="test", revision="main"
    )
    assert spec.source == "huggingface"
    spec.validate_semantic()


def test_dataset_spec_requires_id_unless_local():
    with pytest.raises(ValidationError, match="requires a dataset_id"):
        DatasetSpec(source="huggingface", dataset_id=None)

    spec_local = DatasetSpec(source="local", dataset_id=None, data_dir="/tmp/data")
    spec_local.validate_semantic()  # Should pass


def test_task_spec():
    dataset = DatasetSpec(source="local", data_dir="/tmp")
    spec = TaskSpec(
        task_id="math_eval",
        dataset=dataset,
        default_extractor_chain=ExtractorChainSpec(extractors=["json_extractor"]),
        default_metrics=["exact_match"],
    )
    assert spec.task_id == "math_eval"
    spec.validate_semantic()


def test_task_spec_validation():
    dataset = DatasetSpec(source="local", data_dir="/tmp")
    with pytest.raises(ValidationError, match="must define at least one metric"):
        TaskSpec(task_id="bad", dataset=dataset, default_metrics=[])


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
