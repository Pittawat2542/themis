from __future__ import annotations

from themis.comparison import compare_runs
from themis.experiment import storage as experiment_storage
from tests.factories import make_evaluation_record, make_record


def test_compare_runs_handles_metric_score_lists(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)

    for run_id, value in [("run-a", 1.0), ("run-b", 0.0)]:
        storage.start_run(run_id, experiment_id="default")
        record = make_record(sample_id="s1")
        eval_record = make_evaluation_record(sample_id="s1", metric_name="ExactMatch", value=value)
        storage.append_record(run_id, record, cache_key=experiment_storage.task_cache_key(record.task))
        storage.append_evaluation(run_id, record, eval_record)

    report = compare_runs(run_ids=["run-a", "run-b"], storage_path=tmp_path)

    assert "ExactMatch" in report.metrics
    assert len(report.pairwise_results) == 1
