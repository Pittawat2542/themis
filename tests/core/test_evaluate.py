from __future__ import annotations

from themis import Experiment, InMemoryRunStore, RunStatus, evaluate
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def test_evaluate_runs_one_off_experiment_with_same_run_id_as_explicit_experiment() -> None:
    datasets = [
        Dataset(
            dataset_id="dataset-1",
            cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            revision="r1",
        )
    ]
    explicit = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=datasets,
        seeds=[7],
    )
    store = InMemoryRunStore()

    result = evaluate(
        model="builtin/demo_generator",
        data=datasets,
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
        seeds=[7],
        store=store,
    )

    assert result.status is RunStatus.COMPLETED
    assert result.run_id == explicit.compile().run_id


def test_evaluate_shorthand_supports_workflow_backed_metrics() -> None:
    store = InMemoryRunStore()

    result = evaluate(
        model="builtin/demo_generator",
        data=[
            {
                "case_id": "case-1",
                "input": {"question": "2+2"},
                "expected_output": {"answer": "4"},
            }
        ],
        metric="builtin/llm_rubric",
        parser="builtin/json_identity",
        judge="builtin/demo_judge",
        workflow_overrides={"rubric": "pass if the answer is correct"},
        storage=StorageConfig(store="memory"),
        store=store,
        seeds=[7],
    )

    assert result.status is RunStatus.COMPLETED
    stored = store.resume(result.run_id)

    assert stored is not None
    execution = stored.execution_state.case_states["case-1"].evaluation_executions["builtin/llm_rubric"]
    assert execution.status == "completed"
