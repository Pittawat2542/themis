import pytest
from themis.orchestration.trial_planner import TrialPlanner
from themis.specs.experiment import (
    DataItemContext,
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ItemSamplingSpec,
    PromptTemplateSpec,
)
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec
from themis.errors.exceptions import SpecValidationError
from themis.registry.plugin_registry import PluginRegistry


def test_trial_planner_unrolling():
    models = [
        ModelSpec(model_id="gpt-4", provider="openai"),
        ModelSpec(model_id="claude-3", provider="anthropic"),
    ]

    # Needs two tasks. Let's say one has 3 items, one has 2 items.
    # The dataset adapter is not implemented yet, so we mock it.
    tasks = [
        TaskSpec(
            task_id="t1",
            dataset=DatasetSpec(source="mock", dataset_id="1"),
            default_metrics=["em"],
        ),
        TaskSpec(
            task_id="t2",
            dataset=DatasetSpec(source="mock", dataset_id="2"),
            default_metrics=["em"],
        ),
    ]

    prompts = [
        PromptTemplateSpec(messages=[{"role": "user", "content": "1"}]),
        PromptTemplateSpec(messages=[{"role": "user", "content": "2"}]),
    ]

    inference_grid = [
        InferenceParamsSpec(temperature=0.0),
        InferenceParamsSpec(temperature=0.5),
    ]

    experiment = ExperimentSpec(
        models=models,
        tasks=tasks,
        prompt_templates=prompts,
        inference_grid=InferenceGridSpec(
            params=inference_grid,
            overrides={"max_tokens": [64, 128]},
        ),
        num_samples=1,
        item_sampling=ItemSamplingSpec(kind="subset", count=2),
    )

    # Mocking the dataset loader
    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            if task.task_id == "t1":
                return [
                    {"item_id": "item_1_1", "question": "1 + 1"},
                    {"item_id": "item_1_2", "question": "2 + 2"},
                    {"item_id": "item_1_3", "question": "3 + 3"},
                ]
            if task.task_id == "t2":
                return [
                    {"item_id": "item_2_1", "question": "4 + 4"},
                    {"item_id": "item_2_2", "question": "5 + 5"},
                ]
            return []

    planner = TrialPlanner(dataset_loader=MockDatasetLoader())
    planned_trials = planner.plan_experiment(experiment)

    # Total trials = models (2) * prompts (2) * inferred params (4) * sampled items per task (2 + 2)
    assert len(planned_trials) == 2 * 2 * 4 * 4
    assert all(
        isinstance(planned_trial.dataset_context, DataItemContext)
        for planned_trial in planned_trials
    )
    assert all(
        planned_trial.trial_spec.item_id == planned_trial.dataset_context.item_id
        for planned_trial in planned_trials
    )

    # Every trial generated must be unique
    trial_ids = {planned_trial.trial_spec.trial_id for planned_trial in planned_trials}
    assert len(trial_ids) == 64

    # Verify deterministic trial IDs
    planned_trials_again = planner.plan_experiment(experiment)
    assert [planned_trial.trial_spec.trial_id for planned_trial in planned_trials] == [
        planned_trial.trial_spec.trial_id for planned_trial in planned_trials_again
    ]
    assert [
        planned_trial.dataset_context.item_id for planned_trial in planned_trials
    ] == [
        planned_trial.dataset_context.item_id for planned_trial in planned_trials_again
    ]


def test_trial_planner_missing_dataset_loader():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source="mock", dataset_id="1"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )
    planner = TrialPlanner(dataset_loader=None)
    with pytest.raises(SpecValidationError) as exc:
        planner.plan_experiment(experiment)
    assert "no dataset_loader was provided" in str(exc.value)


def test_trial_planner_validates_trial_compatibility_before_execution():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source="mock", dataset_id="1"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": "6 * 7"}]

    planner = TrialPlanner(
        dataset_loader=MockDatasetLoader(), registry=PluginRegistry()
    )
    with pytest.raises(
        SpecValidationError, match="Provider 'openai' is not registered"
    ):
        planner.plan_experiment(experiment)


def test_trial_planner_rejects_non_json_safe_dataset_items():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source="mock", dataset_id="1"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": object()}]

    planner = TrialPlanner(dataset_loader=MockDatasetLoader())
    with pytest.raises(SpecValidationError, match="dataset item"):
        planner.plan_experiment(experiment)
