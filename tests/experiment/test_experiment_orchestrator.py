import json

import pytest

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline as evaluation_pipeline
from themis.experiment import orchestrator, storage as experiment_storage
from themis.generation import plan as generation_plan
from themis.generation import templates


def make_plan():
    template = templates.PromptTemplate(
        name="qa",
        template="Answer the capital of {topic}.",
    )
    sampling = core_entities.SamplingConfig(temperature=0.5, top_p=0.9, max_tokens=64)
    model_spec = core_entities.ModelSpec(identifier="gpt-4o-mini", provider="fake")
    return generation_plan.GenerationPlan(
        templates=[template],
        models=[model_spec],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field="expected",
        metadata_fields=("subject",),
        context_builder=lambda row: {"topic": row["topic"]},
    )


class EchoRunner:
    def __init__(
        self, answers_by_sample_id: dict[str, str], *, fail_on: set[str] | None = None
    ):
        self.answers_by_sample_id = answers_by_sample_id
        self.fail_on = fail_on or set()
        self.seen_requests: list[core_entities.GenerationTask] = []

    def run(self, requests):
        for task in requests:
            sample_id = task.metadata["dataset_id"]
            self.seen_requests.append(task)
            if sample_id in self.fail_on:
                yield core_entities.GenerationRecord(
                    task=task,
                    output=None,
                    error=core_entities.ModelError(
                        message="model failure", kind="model_error"
                    ),
                    metrics={},
                )
                continue

            answer = self.answers_by_sample_id[sample_id]
            yield core_entities.GenerationRecord(
                task=task,
                output=core_entities.ModelOutput(text=json.dumps({"answer": answer})),
                error=None,
                metrics={},
            )


def make_pipeline():
    extractor = extractors.JsonFieldExtractor(field_path="answer")
    metric = metrics.ExactMatch()
    return evaluation_pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])


class CountingPipeline(evaluation_pipeline.EvaluationPipeline):
    def __init__(self, counter: list[int]):
        super().__init__(
            extractor=extractors.JsonFieldExtractor(field_path="answer"),
            metrics=[metrics.ExactMatch()],
        )
        self.counter = counter

    def evaluate(self, records):  # type: ignore[override]
        self.counter[0] += 1
        return super().evaluate(records)


def make_reference_free_pipeline():
    extractor = extractors.IdentityExtractor()
    metric = metrics.ResponseLength()
    return evaluation_pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])


def build_dataset():
    return [
        {"id": "sample-1", "topic": "France", "expected": "Paris", "subject": "geo"},
        {"id": "sample-2", "topic": "Germany", "expected": "Berlin", "subject": "geo"},
    ]


def test_experiment_run_produces_generation_and_evaluation_reports():
    dataset = build_dataset()
    answers = {row["id"]: row["expected"] for row in dataset}
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    plan = make_plan()
    eval_pipeline = make_pipeline()

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    report = experiment_runner.run(dataset)

    assert len(runner_impl.seen_requests) == len(dataset)
    assert len(report.generation_results) == len(dataset)
    assert report.metadata["total_samples"] == 2
    assert report.metadata["successful_generations"] == 2
    exact_report = report.evaluation_report.metrics["ExactMatch"]
    assert exact_report.count == 2
    assert exact_report.mean == pytest.approx(1.0)


def test_experiment_can_limit_dataset_and_emit_callbacks():
    dataset = build_dataset() + [
        {"id": "sample-3", "topic": "Spain", "expected": "Madrid", "subject": "geo"}
    ]
    answers = {row["id"]: row["expected"] for row in dataset}
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    plan = make_plan()
    eval_pipeline = make_pipeline()

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    seen_ids: list[str] = []

    report = experiment_runner.run(
        dataset,
        max_samples=2,
        on_result=lambda result: seen_ids.append(result.task.metadata["dataset_id"]),
    )

    assert len(report.generation_results) == 2
    assert seen_ids == ["sample-1", "sample-2"]


def test_experiment_records_generation_failures_and_still_evaluates_successes():
    dataset = build_dataset()
    answers = {row["id"]: row["expected"] for row in dataset}
    runner_impl = EchoRunner(answers_by_sample_id=answers, fail_on={"sample-2"})
    plan = make_plan()
    eval_pipeline = make_pipeline()

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    report = experiment_runner.run(dataset)

    assert report.metadata["failed_generations"] == 1
    assert len(report.generation_results) == 2
    exact = report.evaluation_report.metrics["ExactMatch"]
    assert exact.count == 1
    assert exact.mean == 1.0
    assert report.failures[0].sample_id == "sample-2"


def test_experiment_metadata_reports_evaluation_failures():
    dataset = build_dataset()
    answers = {row["id"]: row["expected"] for row in dataset}
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    plan = make_plan()
    bad_pipeline = evaluation_pipeline.EvaluationPipeline(
        extractor=extractors.JsonFieldExtractor(field_path="answer.value"),
        metrics=[metrics.ExactMatch()],
    )
    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=bad_pipeline,
    )

    report = experiment_runner.run(dataset)

    exact = report.evaluation_report.metrics["ExactMatch"]
    assert exact.count == 0
    assert exact.mean == 0.0
    assert report.metadata["evaluation_failures"] == len(dataset) + len(
        report.evaluation_report.failures
    )


def test_experiment_supports_reference_free_metrics():
    dataset = [
        {"id": "sample-1", "topic": "France", "expected": None, "subject": "geo"},
        {"id": "sample-2", "topic": "Germany", "expected": None, "subject": "geo"},
    ]
    answers = {row["id"]: "Paris" for row in dataset}
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    plan = make_plan()
    eval_pipeline = make_reference_free_pipeline()

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    report = experiment_runner.run(dataset)

    length_metric = report.evaluation_report.metrics["ResponseLength"]
    assert length_metric.count == 2


def test_experiment_skips_evaluation_when_cached(tmp_path):
    dataset = build_dataset()
    answers = {row["id"]: row["expected"] for row in dataset}
    storage = experiment_storage.ExperimentStorage(tmp_path / "cache")
    plan = make_plan()
    counter = [0]
    eval_pipeline = CountingPipeline(counter)

    runner_impl = EchoRunner(answers_by_sample_id=answers)
    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
        storage=storage,
    )

    experiment_runner.run(dataset, run_id="cached", resume=False)
    assert counter[0] == 1

    counter[0] = 0
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=CountingPipeline(counter),
        storage=storage,
    )

    experiment_runner.run(dataset=None, run_id="cached", resume=True)
    assert counter[0] == 0


def test_experiment_evaluates_in_configured_batches():
    dataset = build_dataset() + [
        {"id": "sample-3", "topic": "Spain", "expected": "Madrid", "subject": "geo"},
        {"id": "sample-4", "topic": "Italy", "expected": "Rome", "subject": "geo"},
        {"id": "sample-5", "topic": "Japan", "expected": "Tokyo", "subject": "geo"},
    ]
    answers = {row["id"]: row["expected"] for row in dataset}
    plan = make_plan()
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    counter = [0]
    eval_pipeline = CountingPipeline(counter)

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    report = experiment_runner.run(
        dataset,
        resume=False,
        evaluation_batch_size=2,
    )

    # 5 records with batch size 2 -> 3 evaluation calls
    assert counter[0] == 3
    assert report.evaluation_report.metrics["ExactMatch"].count == 5


def test_experiment_bounded_memory_mode_limits_report_records():
    dataset = build_dataset() + [
        {"id": "sample-3", "topic": "Spain", "expected": "Madrid", "subject": "geo"},
        {"id": "sample-4", "topic": "Italy", "expected": "Rome", "subject": "geo"},
        {"id": "sample-5", "topic": "Japan", "expected": "Tokyo", "subject": "geo"},
    ]
    answers = {row["id"]: row["expected"] for row in dataset}
    plan = make_plan()
    runner_impl = EchoRunner(answers_by_sample_id=answers)
    eval_pipeline = make_pipeline()

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    report = experiment_runner.run(
        dataset,
        resume=False,
        max_records_in_memory=2,
    )

    assert report.metadata["total_samples"] == 5
    assert report.metadata["generation_records_retained"] == 2
    assert report.metadata["generation_records_dropped"] == 3
    assert report.metadata["evaluation_records_retained"] == 2
    assert report.metadata["evaluation_records_dropped"] == 3
    assert report.metadata["successful_generations"] == 5
    assert len(report.generation_results) == 2
    assert len(report.evaluation_report.records) == 2
    assert report.evaluation_report.metrics["ExactMatch"].count == 5


def test_experiment_bounded_memory_mode_rejects_invalid_limit():
    plan = make_plan()
    runner_impl = EchoRunner(answers_by_sample_id={})
    eval_pipeline = make_pipeline()

    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
    )

    with pytest.raises(ValueError, match="max_records_in_memory"):
        experiment_runner.run(
            [{"id": "sample-1", "topic": "France", "expected": "Paris", "subject": "geo"}],
            max_records_in_memory=0,
        )


def test_experiment_can_resume_from_storage(tmp_path):
    dataset = build_dataset()
    answers = {row["id"]: row["expected"] for row in dataset}
    storage = experiment_storage.ExperimentStorage(tmp_path / "cache")
    plan = make_plan()
    eval_pipeline = make_pipeline()

    runner_impl = EchoRunner(answers_by_sample_id=answers)
    experiment_runner = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner_impl,
        evaluation_pipeline=eval_pipeline,
        storage=storage,
    )

    run_id = "math-demo"
    experiment_runner.run(dataset, run_id=run_id, resume=False)
    assert len(runner_impl.seen_requests) == len(dataset)

    class NoWorkRunner(EchoRunner):
        def run(self, requests):  # type: ignore[override]
            captured = list(requests)
            assert captured == []
            return iter([])

    noop_runner = NoWorkRunner(answers_by_sample_id=answers)
    resume_orchestrator = orchestrator.ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=noop_runner,
        evaluation_pipeline=eval_pipeline,
        storage=storage,
    )

    report = resume_orchestrator.run(dataset=None, run_id=run_id, resume=True)

    assert report.metadata["successful_generations"] == len(dataset)
    assert noop_runner.seen_requests == []
