import pytest
from pydantic import ValidationError

from themis.specs.experiment import (
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ItemSamplingSpec,
    ProjectSpec,
    PromptMessage,
    StorageSpec,
    TrialSpec,
    ExperimentSpec,
    PromptTemplateSpec,
    RuntimeContext,
)
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    ModelSpec,
    TaskSpec,
)


def _make_model():
    return ModelSpec(model_id="gpt-4", provider="openai")


def _make_task():
    return TaskSpec(
        task_id="t1",
        dataset=DatasetSpec(source="memory"),
        default_extractor_chain=ExtractorChainSpec(extractors=["json"]),
        default_metrics=["em"],
    )


def test_inference_params():
    params = InferenceParamsSpec(
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        response_format="json",
    )
    assert params.temperature == 0.5
    assert params.top_p == 0.9
    assert params.max_tokens == 100
    assert params.response_format == "json"


def test_prompt_template_spec():
    spec = PromptTemplateSpec(messages=[{"role": "user", "content": "Hello {name}"}])
    assert spec.messages[0].content == "Hello {name}"


def test_prompt_template_spec_rejects_unknown_message_keys():
    with pytest.raises(ValidationError):
        PromptTemplateSpec(
            messages=[{"role": "user", "content": "Hello", "extra": "bad"}]
        )


def test_prompt_message_requires_role_and_content():
    message = PromptMessage(role="assistant", content="Done")

    assert message.role == "assistant"
    assert message.content == "Done"

    with pytest.raises(ValidationError):
        PromptMessage.model_validate({"foo": "bar"})


def test_trial_spec_validation():
    trial = TrialSpec(
        trial_id="t_1",
        model=_make_model(),
        task=_make_task(),
        item_id="item_0",
        prompt=PromptTemplateSpec(messages=[{"role": "user", "content": "Q: {q}"}]),
        params=InferenceParamsSpec(),
        candidate_count=5,
    )
    trial.validate_semantic()

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        TrialSpec(
            trial_id="t_2",
            model=_make_model(),
            task=_make_task(),
            item_id="item_0",
            prompt=PromptTemplateSpec(messages=[]),
            params=InferenceParamsSpec(),
            candidate_count=0,
        )


def test_experiment_spec():
    spec = ExperimentSpec(
        models=[_make_model()],
        tasks=[_make_task()],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(temperature=0.0)]),
        item_sampling=ItemSamplingSpec(kind="all"),
    )
    assert len(spec.models) == 1
    spec.validate_semantic()


def test_experiment_spec_empty_validation():
    with pytest.raises(ValidationError, match="must have at least one model"):
        ExperimentSpec(
            models=[],
            tasks=[_make_task()],
            prompt_templates=[PromptTemplateSpec(messages=[])],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(temperature=0.0)]
            ),
        )


def test_project_spec():
    proj = ProjectSpec(
        project_name="eval_run_1",
        researcher_id="researcher_1",
        global_seed=42,
        storage=StorageSpec(backend="sqlite_blob", root_dir="./runs/eval_run_1"),
        execution_policy=ExecutionPolicySpec(),
    )
    assert proj.project_name == "eval_run_1"


def test_runtime_context_defaults_and_immutability():
    runtime = RuntimeContext()

    assert runtime.secrets == {}
    assert runtime.environment == {}
    assert runtime.run_labels == {}

    with pytest.raises(Exception):
        runtime.environment = {"suite": "tests"}


def test_inference_grid_spec_tracks_param_bases_and_overrides():
    spec = InferenceGridSpec(
        params=[InferenceParamsSpec(temperature=0.0, max_tokens=32)],
        overrides={"max_tokens": [32, 64], "top_p": [0.9]},
    )

    assert len(spec.params) == 1
    assert spec.overrides["max_tokens"] == [32, 64]


def test_item_sampling_subset_requires_count():
    with pytest.raises(ValidationError, match="requires a positive count"):
        ItemSamplingSpec(kind="subset")


def test_item_sampling_stratified_requires_strata_field():
    with pytest.raises(ValidationError, match="requires strata_field"):
        ItemSamplingSpec(kind="stratified", count=2)


def test_project_spec_rejects_string_seed_under_strict_mode():
    with pytest.raises(ValidationError):
        ProjectSpec(
            project_name="eval_run_1",
            researcher_id="researcher_1",
            global_seed="42",
            storage=StorageSpec(backend="sqlite_blob", root_dir="./runs/eval_run_1"),
            execution_policy=ExecutionPolicySpec(),
        )


def test_inference_params_rejects_string_numbers_under_strict_mode():
    with pytest.raises(ValidationError):
        InferenceParamsSpec(max_tokens="100")


def test_runtime_context_resume_is_typed():
    with pytest.raises(ValidationError):
        RuntimeContext.model_validate({"resume": object()})
