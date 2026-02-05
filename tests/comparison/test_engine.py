from __future__ import annotations

import pytest

from themis.core import entities as core_entities
from themis.comparison.engine import ComparisonEngine
from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest, StatisticalTestResult
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


def test_compare_runs_requires_overlapping_sample_ids(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)

    # Run A evaluated on sample set s1,s2
    storage.start_run("run-a", experiment_id="default")
    for sample_id, value in [("s1", 1.0), ("s2", 0.0)]:
        record = make_record(sample_id=sample_id)
        eval_record = make_evaluation_record(
            sample_id=sample_id, metric_name="ExactMatch", value=value
        )
        storage.append_record(
            "run-a",
            record,
            cache_key=experiment_storage.task_cache_key(record.task),
        )
        storage.append_evaluation("run-a", record, eval_record)

    # Run B evaluated on a disjoint sample set t1,t2
    storage.start_run("run-b", experiment_id="default")
    for sample_id, value in [("t1", 0.0), ("t2", 1.0)]:
        record = make_record(sample_id=sample_id)
        eval_record = make_evaluation_record(
            sample_id=sample_id, metric_name="ExactMatch", value=value
        )
        storage.append_record(
            "run-b",
            record,
            cache_key=experiment_storage.task_cache_key(record.task),
        )
        storage.append_evaluation("run-b", record, eval_record)

    with pytest.raises(ValueError, match="overlapping sample_ids"):
        compare_runs(
            run_ids=["run-a", "run-b"],
            storage_path=tmp_path,
            statistical_test=StatisticalTest.T_TEST,
        )


def test_compare_runs_applies_holm_bonferroni_by_default(tmp_path, monkeypatch):
    storage = experiment_storage.ExperimentStorage(tmp_path)

    for run_id, metric_a, metric_b in [("run-a", 1.0, 1.0), ("run-b", 0.0, 0.0)]:
        storage.start_run(run_id, experiment_id="default")
        record = make_record(sample_id="s1")
        eval_record = core_entities.EvaluationRecord(
            sample_id="s1",
            scores=[
                core_entities.MetricScore(metric_name="MetricA", value=metric_a),
                core_entities.MetricScore(metric_name="MetricB", value=metric_b),
            ],
        )
        storage.append_record(
            run_id,
            record,
            cache_key=experiment_storage.task_cache_key(record.task),
        )
        storage.append_evaluation(run_id, record, eval_record)

    p_values = iter([0.01, 0.08])  # First survives Holm, second is rejected

    def _fake_t_test(_a, _b, alpha=0.05, paired=True):
        return StatisticalTestResult(
            test_name="t-test (paired)",
            statistic=2.0,
            p_value=next(p_values),
            significant=True,
        )

    monkeypatch.setattr("themis.comparison.engine.statistics.t_test", _fake_t_test)

    engine = ComparisonEngine(
        storage=storage,
        statistical_test=StatisticalTest.T_TEST,
        alpha=0.05,
    )
    report = engine.compare_runs(["run-a", "run-b"], metrics=["MetricA", "MetricB"])
    by_metric = {result.metric_name: result for result in report.pairwise_results}

    assert by_metric["MetricA"].corrected_significant is True
    assert by_metric["MetricA"].winner == "run-a"
    assert by_metric["MetricB"].corrected_significant is False
    assert by_metric["MetricB"].winner == "tie"
    assert report.metadata["multiple_comparison_correction"] == "holm-bonferroni"
    assert report.metadata["n_hypotheses_corrected"] == 2


def test_compare_runs_holm_correction_respects_engine_alpha(tmp_path, monkeypatch):
    storage = experiment_storage.ExperimentStorage(tmp_path)

    for run_id, value in [("run-a", 1.0), ("run-b", 0.0)]:
        storage.start_run(run_id, experiment_id="default")
        record = make_record(sample_id="s1")
        eval_record = make_evaluation_record(
            sample_id="s1", metric_name="ExactMatch", value=value
        )
        storage.append_record(
            run_id,
            record,
            cache_key=experiment_storage.task_cache_key(record.task),
        )
        storage.append_evaluation(run_id, record, eval_record)

    def _fake_t_test(_a, _b, alpha=0.05, paired=True):
        return StatisticalTestResult(
            test_name="t-test (paired)",
            statistic=2.0,
            p_value=0.03,
            significant=True,
        )

    monkeypatch.setattr("themis.comparison.engine.statistics.t_test", _fake_t_test)

    engine = ComparisonEngine(
        storage=storage,
        statistical_test=StatisticalTest.T_TEST,
        alpha=0.01,
    )
    report = engine.compare_runs(["run-a", "run-b"], metrics=["ExactMatch"])
    result = report.pairwise_results[0]

    assert result.corrected_significant is False
    assert result.winner == "tie"
