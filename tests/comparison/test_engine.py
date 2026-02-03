from __future__ import annotations

from themis.comparison import compare_runs
from themis.core import entities as core_entities
from themis.experiment import storage as experiment_storage


def _make_record(sample_id: str, value: float) -> tuple[core_entities.GenerationRecord, core_entities.EvaluationRecord]:
    prompt_spec = core_entities.PromptSpec(name="t", template="Q")
    prompt = core_entities.PromptRender(spec=prompt_spec, text="Q")
    model = core_entities.ModelSpec(identifier="model-x", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    task = core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": sample_id},
    )
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="ok"),
        error=None,
    )
    eval_record = core_entities.EvaluationRecord(
        sample_id=sample_id,
        scores=[
            core_entities.MetricScore(metric_name="ExactMatch", value=value),
        ],
    )
    return record, eval_record


def test_compare_runs_handles_metric_score_lists(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)

    for run_id, value in [("run-a", 1.0), ("run-b", 0.0)]:
        storage.start_run(run_id, experiment_id="default")
        record, eval_record = _make_record(sample_id="s1", value=value)
        storage.append_record(run_id, record, cache_key=experiment_storage.task_cache_key(record.task))
        storage.append_evaluation(run_id, record, eval_record)

    report = compare_runs(run_ids=["run-a", "run-b"], storage_path=tmp_path)

    assert "ExactMatch" in report.metrics
    assert len(report.pairwise_results) == 1
