import pytest
from themis.benchmark.query import DatasetQuerySpec
from themis.orchestration.trial_planner import TrialPlanner
from themis.specs.experiment import (
    DataItemContext,
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ItemSamplingSpec,
    PromptTemplateSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    GenerationSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.errors import SpecValidationError
from themis.registry.plugin_registry import PluginRegistry
from themis.types.enums import SamplingKind, DatasetSource, RunStage


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
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(extractors=["json"]),
                )
            ],
            evaluations=[
                EvaluationSpec(name="judge", transform="json", metrics=["em"])
            ],
        ),
        TaskSpec(
            task_id="t2",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(extractors=["json"]),
                )
            ],
            evaluations=[
                EvaluationSpec(name="judge", transform="json", metrics=["em"])
            ],
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
        item_sampling=ItemSamplingSpec.subset(2),
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
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
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


def test_trial_planner_rejects_compatibility_checker_override():
    with pytest.raises(TypeError, match="compatibility_checker"):
        TrialPlanner(compatibility_checker=object())


def test_trial_planner_validates_trial_compatibility_before_execution():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
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


def test_trial_planner_validates_output_transform_compatibility_before_execution():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="json",
                        extractor_chain=ExtractorChainSpec(
                            extractors=["missing_extractor"]
                        ),
                    )
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": "6 * 7"}]

    class DummyInferenceEngine:
        def infer(self, trial, dataset_context, runtime_context):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_inference_engine("openai", DummyInferenceEngine())
    planner = TrialPlanner(dataset_loader=MockDatasetLoader(), registry=registry)

    with pytest.raises(
        SpecValidationError, match="Extractor 'missing_extractor' is not registered"
    ):
        planner.plan_experiment(experiment)


def test_trial_planner_validates_evaluation_compatibility_before_execution():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="json",
                        extractor_chain=ExtractorChainSpec(extractors=["regex"]),
                    )
                ],
                evaluations=[
                    EvaluationSpec(
                        name="judge",
                        transform="json",
                        metrics=["missing_metric"],
                    )
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": "6 * 7"}]

    class DummyInferenceEngine:
        def infer(self, trial, dataset_context, runtime_context):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_inference_engine("openai", DummyInferenceEngine())
    planner = TrialPlanner(dataset_loader=MockDatasetLoader(), registry=registry)

    with pytest.raises(
        SpecValidationError, match="Metric 'missing_metric' is not registered"
    ):
        planner.plan_experiment(experiment)


def test_trial_planner_filters_items_before_sampling():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
        item_sampling=ItemSamplingSpec(
            kind=SamplingKind.SUBSET,
            count=1,
            seed=11,
            item_ids=["item-1", "item-2", "item-3"],
            metadata_filters={"difficulty": "hard"},
        ),
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            del task
            return [
                {
                    "item_id": "item-1",
                    "question": "easy one",
                    "metadata": {"difficulty": "easy"},
                },
                {
                    "item_id": "item-2",
                    "question": "hard one",
                    "metadata": {"difficulty": "hard"},
                },
                {
                    "item_id": "item-3",
                    "question": "hard two",
                    "metadata": {"difficulty": "hard"},
                },
                {
                    "item_id": "item-4",
                    "question": "hard but excluded",
                    "metadata": {"difficulty": "hard"},
                },
            ]

    class DummyInferenceEngine:
        def infer(self, trial, dataset_context, runtime_context):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_inference_engine("openai", DummyInferenceEngine())
    planner = TrialPlanner(dataset_loader=MockDatasetLoader(), registry=registry)

    planned_trials = planner.plan_experiment(experiment)

    assert len(planned_trials) == 1
    assert planned_trials[0].dataset_context.item_id in {"item-2", "item-3"}
    assert planned_trials[0].dataset_context.metadata["difficulty"] == "hard"


def test_trial_planner_rejects_non_json_safe_dataset_items():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
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


def test_trial_hash_changes_when_transforms_or_evaluations_change():
    task_a = TaskSpec(
        task_id="qa",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        generation=GenerationSpec(),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            )
        ],
        evaluations=[EvaluationSpec(name="judge", transform="json", metrics=["em"])],
    )
    task_b = task_a.model_copy(
        update={
            "evaluations": [
                EvaluationSpec(name="judge2", transform="json", metrics=["f1"])
            ]
        }
    )

    assert task_a.spec_hash != task_b.spec_hash


def test_trial_planner_can_validate_generation_stage_without_transform_or_metric_plugins():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="json",
                        extractor_chain=ExtractorChainSpec(
                            extractors=["missing_extractor"]
                        ),
                    )
                ],
                evaluations=[
                    EvaluationSpec(
                        name="judge",
                        transform="json",
                        metrics=["missing_metric"],
                    )
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": "6 * 7"}]

    class DummyInferenceEngine:
        def infer(self, trial, dataset_context, runtime_context):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_inference_engine("openai", DummyInferenceEngine())
    planner = TrialPlanner(dataset_loader=MockDatasetLoader(), registry=registry)

    planned_trials = planner.plan_experiment(
        experiment,
        required_stages={RunStage.GENERATION},
    )

    assert len(planned_trials) == 1


def test_trial_planner_can_validate_transform_only_task_without_provider_plugin():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="imported-model", provider="unregistered")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                output_transforms=[
                    OutputTransformSpec(
                        name="json",
                        extractor_chain=ExtractorChainSpec(extractors=["json"]),
                    )
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": "6 * 7"}]

    class DummyExtractor:
        def extract(self, trial, candidate, config=None):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_extractor("json", DummyExtractor())
    planner = TrialPlanner(dataset_loader=MockDatasetLoader(), registry=registry)

    planned_trials = planner.plan_experiment(
        experiment,
        required_stages={RunStage.TRANSFORM},
    )

    assert len(planned_trials) == 1


def test_trial_planner_can_validate_evaluation_only_task_without_provider_plugin():
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="imported-model", provider="unregistered")],
        tasks=[
            TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                evaluations=[EvaluationSpec(name="judge", metrics=["em"])],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            return [{"item_id": "item-1", "question": "6 * 7"}]

    class DummyMetric:
        def score(self, trial, candidate, context):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_metric("em", DummyMetric())
    planner = TrialPlanner(dataset_loader=MockDatasetLoader(), registry=registry)

    planned_trials = planner.plan_experiment(
        experiment,
        required_stages={RunStage.EVALUATION},
    )

    assert len(planned_trials) == 1


def test_trial_planner_propagates_unexpected_dataset_query_validation_errors(
    monkeypatch,
):
    planner = TrialPlanner()

    def _boom(cls, payload):
        del cls, payload
        raise RuntimeError("unexpected dataset query failure")

    monkeypatch.setattr(DatasetQuerySpec, "model_validate", classmethod(_boom))

    with pytest.raises(RuntimeError, match="unexpected dataset query failure"):
        planner._coerce_dataset_query({"kind": "all"})


def test_item_sampling_classmethods_preserve_sampling_behavior():
    class MockDatasetLoader:
        def load_task_items(self, task: TaskSpec):
            del task
            return [
                {"item_id": "item-1", "bucket": "a"},
                {"item_id": "item-2", "bucket": "a"},
                {"item_id": "item-3", "bucket": "b"},
                {"item_id": "item-4", "bucket": "b"},
            ]

    planner = TrialPlanner(dataset_loader=MockDatasetLoader())
    task = TaskSpec(
        task_id="t1",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        generation=GenerationSpec(),
    )
    items = MockDatasetLoader().load_task_items(task)

    subset_items = planner._sample_items(items, ItemSamplingSpec.subset(2, seed=11))
    assert len(subset_items) == 2
    assert ItemSamplingSpec.subset(2).kind == SamplingKind.SUBSET

    stratified_items = planner._sample_items(
        items,
        ItemSamplingSpec.stratified(2, strata_field="bucket", seed=11),
    )
    assert len(stratified_items) == 2
    assert {item["bucket"] for item in stratified_items} == {"a", "b"}
