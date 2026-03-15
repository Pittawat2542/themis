import pytest
from pydantic import ValidationError

import themis.specs.foundational as foundational
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
from themis.types.enums import DatasetSource


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
    )
    assert spec.task_id == "math_eval"


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
