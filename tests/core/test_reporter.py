from __future__ import annotations

import csv
import json
from io import StringIO
from typing import cast

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import (
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.quickcheck import quickcheck
from themis.core.registry import RegressionPolicy, RunRecord
from themis.core import reporter as reporter_module
from themis.core.reporter import Reporter, snapshot_report
from themis.core.store import RunStore
from themis.core.results import ProjectionCursor
from themis.core.stores.memory import InMemoryRunStore
from tests.release import CURRENT_VERSION


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version=CURRENT_VERSION,
        python_version="3.12.9",
        platform="macos",
    )
    return experiment.compile()


def _snapshot_with_cases(*, seed: int, case_ids: list[str]):
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[
                    Case(
                        case_id=case_id,
                        input={"question": case_id},
                        expected_output={"answer": "4"},
                    )
                    for case_id in case_ids
                ],
            )
        ],
        seeds=[seed],
    )
    return experiment.compile()


def _persist_scored_run(
    store: InMemoryRunStore,
    *,
    seed: int,
    scores: dict[str, float],
    tags: list[str] | None = None,
) -> str:
    snapshot = _snapshot_with_cases(seed=seed, case_ids=list(scores))
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    for index, (case_id, score) in enumerate(scores.items()):
        candidate_id = f"{case_id}-candidate-0"
        store.persist_event(
            GenerationCompletedEvent(
                run_id=snapshot.run_id,
                case_id=case_id,
                dataset_id="dataset-1",
                candidate_id=candidate_id,
                candidate_index=index,
                result={"candidate_id": candidate_id, "final_output": {"answer": "4"}},
            )
        )
        store.persist_event(
            ReductionCompletedEvent(
                run_id=snapshot.run_id,
                case_id=case_id,
                dataset_id="dataset-1",
                candidate_id=f"{case_id}-reduced",
                source_candidate_ids=[candidate_id],
                result={
                    "candidate_id": f"{case_id}-reduced",
                    "source_candidate_ids": [candidate_id],
                    "final_output": {"answer": "4"},
                },
            )
        )
        store.persist_event(
            ParseCompletedEvent(
                run_id=snapshot.run_id,
                case_id=case_id,
                dataset_id="dataset-1",
                candidate_id=f"{case_id}-reduced",
                result={"value": {"answer": "4"}, "format": "json"},
            )
        )
        store.persist_event(
            ScoreCompletedEvent(
                run_id=snapshot.run_id,
                case_id=case_id,
                dataset_id="dataset-1",
                candidate_id=f"{case_id}-reduced",
                metric_id="builtin/exact_match",
                metric_result={"metric_id": "builtin/exact_match", "value": score},
            )
        )
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    if tags:
        store.update_run_record(snapshot.run_id, tags=tags)
    return snapshot.run_id


def _store() -> tuple[InMemoryRunStore, str]:
    store = InMemoryRunStore()
    snapshot = _snapshot()
    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="candidate-1",
            candidate_index=0,
            seed=7,
            result={"candidate_id": "candidate-1", "final_output": {"answer": "4"}},
            result_blob_ref="sha256:generation-1",
        )
    )
    store.persist_event(
        ReductionCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            source_candidate_ids=["candidate-1"],
            result={
                "candidate_id": "case-1-reduced",
                "source_candidate_ids": ["candidate-1"],
                "final_output": {"answer": "4"},
            },
        )
    )
    store.persist_event(
        ParseCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            result={"value": {"answer": "4"}, "format": "json"},
        )
    )
    store.persist_event(
        ScoreCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="builtin/exact_match",
            metric_result={
                "metric_id": "builtin/exact_match",
                "value": 1.0,
                "metadata": {"matched": True},
            },
        )
    )
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    return store, snapshot.run_id


def _mark_benchmark_fresh(store: InMemoryRunStore, run_id: str) -> None:
    store.store_projection_cursor(
        ProjectionCursor(
            run_id=run_id,
            projection_name="benchmark_result",
            event_count=store.count_events(run_id),
        )
    )


def test_reporter_exports_valid_json_markdown_csv_and_latex() -> None:
    store, run_id = _store()
    reporter = Reporter(store)

    exported_json = reporter.export_json(run_id)
    exported_markdown = reporter.export_markdown(run_id)
    exported_csv = reporter.export_csv(run_id)
    exported_latex = reporter.export_latex(run_id)
    score_rows = reporter.score_rows(run_id)
    summary = reporter.summary(run_id)

    parsed_json = json.loads(exported_json)
    csv_rows = list(csv.DictReader(StringIO(exported_csv)))

    assert parsed_json["run_result"]["run_id"] == run_id
    assert parsed_json["snapshot"]["run_id"] == run_id
    assert parsed_json["execution_state"]["run_id"] == run_id
    assert parsed_json["stats_summary"] == summary.model_dump(mode="json")
    assert "# Run Report" in exported_markdown
    assert (
        "| metric_id | count | mean | min | max | ci_lower | ci_upper |"
        in exported_markdown
    )
    assert "builtin/exact_match" in exported_markdown
    assert "## Failures" not in exported_markdown
    assert len(csv_rows) == 1
    assert csv_rows[0]["metric_id"] == "builtin/exact_match"
    assert csv_rows[0]["mean"] == "1.0"
    assert "\\begin{tabular}" in exported_latex
    assert r"\begin{tabular}{lrrrrrr}" in exported_latex
    assert score_rows == [
        {
            "case_id": "case-1",
            "dataset_id": "dataset-1",
            "case_key": "9:dataset-1:case-1",
            "metric_id": "builtin/exact_match",
            "result_type": "scalar",
            "outcome": "correct",
            "value": 1.0,
            "confidence": None,
            "dimensions": {},
            "labels": {},
            "candidate_id": "case-1-reduced",
            "failure_category": None,
            "error_message": None,
            "metadata": {"matched": True},
        }
    ]


def test_reporter_compare_runs_returns_score_claim_with_missing_counts() -> None:
    store = InMemoryRunStore()
    store.initialize()
    baseline_run_id = _persist_scored_run(
        store, seed=7, scores={"case-1": 0.0, "case-2": 1.0}
    )
    candidate_run_id = _persist_scored_run(
        store, seed=11, scores={"case-1": 1.0, "case-3": 1.0}
    )

    report = Reporter(store).compare_runs(baseline_run_id, candidate_run_id)

    assert report.claim_type == "score_claim"
    assert report.evidence_run_ids == [baseline_run_id, candidate_run_id]
    assert report.metrics[0].pairs == 1
    assert report.metrics[0].mean_delta == 1.0
    assert report.matched_pair_count == 1
    assert report.missing_baseline_rows == 1
    assert report.missing_candidate_rows == 1
    assert report.dropped_rows == 2


def test_reporter_suite_coverage_counts_runs_tagged_by_suite() -> None:
    store = InMemoryRunStore()
    store.initialize()
    first_run_id = _persist_scored_run(
        store,
        seed=7,
        scores={"case-1": 1.0},
        tags=["suite:math-core", "benchmark:gsm8k"],
    )
    _persist_scored_run(
        store,
        seed=11,
        scores={"case-1": 1.0},
        tags=["suite:math-core", "benchmark:math500"],
    )

    coverage = Reporter(store).suite_coverage("math-core")

    assert coverage.suite_id == "math-core"
    assert coverage.covered_count == 2
    assert coverage.total_runs == 2
    assert coverage.covered_benchmark_ids == ["gsm8k", "math500"]
    assert first_run_id in coverage.run_ids


def test_default_reporter_is_registered_and_resolved() -> None:
    store, run_id = _store()

    assert "default" in reporter_module.available_reporters()
    reporter = reporter_module.create_reporter("default", store)

    assert isinstance(reporter, Reporter)
    assert reporter.summary(run_id).run_id == run_id


def test_custom_reporter_can_be_registered_and_resolved(monkeypatch) -> None:
    monkeypatch.setattr(
        reporter_module,
        "_REPORTER_BUILDERS",
        dict(reporter_module._REPORTER_BUILDERS),
    )

    class MinimalReporter:
        def __init__(self, store: RunStore) -> None:
            self.store = store

        def export_json(self, run_id: str) -> str:
            return json.dumps(
                {"run_id": run_id, "events": self.store.count_events(run_id)}
            )

    store, run_id = _store()

    def build_minimal_reporter(store: RunStore) -> MinimalReporter:
        return MinimalReporter(store)

    reporter_module.register_reporter("test/minimal", build_minimal_reporter)

    reporter = reporter_module.create_reporter("test/minimal", store)

    assert json.loads(reporter.export_json(run_id)) == {
        "run_id": run_id,
        "events": 6,
    }


def test_reporter_selection_does_not_affect_run_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        reporter_module,
        "_REPORTER_BUILDERS",
        dict(reporter_module._REPORTER_BUILDERS),
    )

    first = _snapshot()

    class IdentityReporter:
        def __init__(self, store: RunStore) -> None:
            self.store = store

        def export_json(self, run_id: str) -> str:
            return json.dumps({"run_id": run_id})

    def build_identity_reporter(store: RunStore) -> IdentityReporter:
        return IdentityReporter(store)

    reporter_module.register_reporter("test/identity", build_identity_reporter)
    second = _snapshot()

    assert reporter_module.create_reporter("test/identity", InMemoryRunStore())
    assert first.run_id == second.run_id


def test_reporter_escapes_latex_special_characters() -> None:
    store, run_id = _store()
    store._projections[(run_id, "benchmark_result")] = {
        "run_id": run_id,
        "dataset_ids": ["data_set%1"],
        "metric_ids": ["metric_^~#"],
        "total_cases": 1,
        "completed_cases": 1,
        "failed_cases": 0,
        "score_rows": [
            {
                "case_id": r"case_1%&${}\path",
                "metric_id": "metric_^~#",
                "outcome": "correct",
                "value": 1.0,
                "candidate_id": None,
            }
        ],
        "metric_means": {"metric_^~#": 1.0},
        "outcome_counts": {"metric_^~#": {"correct": 1}},
        "error_counts": {},
    }
    _mark_benchmark_fresh(store, run_id)
    reporter = Reporter(store)

    exported_latex = reporter.export_latex(run_id)

    assert r"metric\_\textasciicircum{}\textasciitilde{}\#" in exported_latex


def test_reporter_markdown_includes_failure_section_only_for_error_rows() -> None:
    store, run_id = _store()
    store._projections[(run_id, "benchmark_result")] = {
        "run_id": run_id,
        "dataset_ids": ["dataset-1"],
        "metric_ids": ["builtin/exact_match"],
        "total_cases": 1,
        "completed_cases": 0,
        "failed_cases": 1,
        "score_rows": [
            {
                "case_id": "case-1",
                "dataset_id": "dataset-1",
                "case_key": "9:dataset-1:case-1",
                "metric_id": "builtin/exact_match",
                "outcome": "error",
                "value": None,
                "candidate_id": "case-1-reduced",
                "failure_category": "parse_failure",
                "error_message": "bad parse",
            }
        ],
        "metric_means": {},
        "outcome_counts": {"builtin/exact_match": {"error": 1}},
        "error_counts": {"builtin/exact_match": {"parse_failure": 1}},
    }
    _mark_benchmark_fresh(store, run_id)
    reporter = Reporter(store)

    exported_markdown = reporter.export_markdown(run_id)

    assert "## Failures" in exported_markdown
    assert (
        "| case_id | metric_id | failure_category | error_message |"
        in exported_markdown
    )
    assert (
        "| case-1 | builtin/exact_match | parse_failure | bad parse |"
        in exported_markdown
    )


def test_reporter_builds_failure_slices_from_error_rows() -> None:
    store, run_id = _store()
    store._projections[(run_id, "benchmark_result")] = {
        "run_id": run_id,
        "dataset_ids": ["dataset-1"],
        "metric_ids": ["builtin/exact_match", "builtin/f1"],
        "total_cases": 2,
        "completed_cases": 0,
        "failed_cases": 2,
        "score_rows": [
            {
                "case_id": "case-1",
                "dataset_id": "dataset-1",
                "case_key": "9:dataset-1:case-1",
                "metric_id": "builtin/exact_match",
                "outcome": "error",
                "value": None,
                "candidate_id": "case-1-reduced",
                "failure_category": "parse_failure",
                "error_message": "bad parse",
                "metadata": {"slice": "math"},
            },
            {
                "case_id": "case-2",
                "dataset_id": "dataset-1",
                "case_key": "9:dataset-1:case-2",
                "metric_id": "builtin/f1",
                "outcome": "error",
                "value": None,
                "candidate_id": "case-2-reduced",
                "failure_category": "provider_failure",
                "error_message": "timeout",
                "metadata": {"slice": "math"},
            },
        ],
        "metric_means": {},
        "outcome_counts": {
            "builtin/exact_match": {"error": 1},
            "builtin/f1": {"error": 1},
        },
        "error_counts": {
            "builtin/exact_match": {"parse_failure": 1},
            "builtin/f1": {"provider_failure": 1},
        },
    }

    _mark_benchmark_fresh(store, run_id)
    slices = Reporter(store).failure_slices(run_id)

    assert [item.model_dump() for item in slices.slices] == [
        {
            "dimension": "category",
            "value": "parse_failure",
            "count": 1,
            "case_keys": ["9:dataset-1:case-1"],
        },
        {
            "dimension": "category",
            "value": "provider_failure",
            "count": 1,
            "case_keys": ["9:dataset-1:case-2"],
        },
        {
            "dimension": "dataset",
            "value": "dataset-1",
            "count": 2,
            "case_keys": ["9:dataset-1:case-1", "9:dataset-1:case-2"],
        },
        {
            "dimension": "metadata.slice",
            "value": "math",
            "count": 2,
            "case_keys": ["9:dataset-1:case-1", "9:dataset-1:case-2"],
        },
        {
            "dimension": "metric",
            "value": "builtin/exact_match",
            "count": 1,
            "case_keys": ["9:dataset-1:case-1"],
        },
        {
            "dimension": "metric",
            "value": "builtin/f1",
            "count": 1,
            "case_keys": ["9:dataset-1:case-2"],
        },
    ]


def test_reporter_reliability_summarizes_confidence_calibration() -> None:
    store, run_id = _store()
    store._projections[(run_id, "benchmark_result")] = {
        "run_id": run_id,
        "dataset_ids": ["dataset-1"],
        "metric_ids": ["metric/confidence"],
        "total_cases": 2,
        "completed_cases": 2,
        "failed_cases": 0,
        "score_rows": [
            {
                "case_id": "case-1",
                "dataset_id": "dataset-1",
                "case_key": "9:dataset-1:case-1",
                "metric_id": "metric/confidence",
                "outcome": "correct",
                "value": 1.0,
                "confidence": 0.8,
                "candidate_id": "case-1-reduced",
            },
            {
                "case_id": "case-2",
                "dataset_id": "dataset-1",
                "case_key": "9:dataset-1:case-2",
                "metric_id": "metric/confidence",
                "outcome": "incorrect",
                "value": 0.0,
                "confidence": 0.3,
                "candidate_id": "case-2-reduced",
            },
        ],
        "metric_means": {"metric/confidence": 0.5},
        "outcome_counts": {"metric/confidence": {"correct": 1, "incorrect": 1}},
        "error_counts": {},
    }

    _mark_benchmark_fresh(store, run_id)
    summary = Reporter(store).reliability(run_id)

    assert [metric.model_dump() for metric in summary.metrics] == [
        {
            "metric_id": "metric/confidence",
            "result_type": "calibration",
            "value": 0.25,
            "dimensions": {"sample_count": 2.0},
            "labels": {},
            "confidence": None,
            "metadata": {},
        }
    ]


def test_reporter_builds_metric_trends_from_run_registry() -> None:
    store, run_id = _store()
    candidate_run_id = "candidate-run"
    baseline = store.get_run_record(run_id)
    assert baseline is not None
    store._run_records[candidate_run_id] = RunRecord(
        run_id=candidate_run_id,
        status="completed",
        dataset_source_ids=["dataset-1"],
        dataset_fingerprints=["fingerprint-1"],
        metric_ids=["builtin/exact_match"],
        baseline_label="candidate",
    )
    store._projections[(candidate_run_id, "benchmark_result")] = {
        "run_id": candidate_run_id,
        "dataset_ids": ["dataset-1"],
        "metric_ids": ["builtin/exact_match"],
        "total_cases": 1,
        "completed_cases": 1,
        "failed_cases": 0,
        "score_rows": [],
        "metric_means": {"builtin/exact_match": 0.75},
        "outcome_counts": {},
        "error_counts": {},
    }

    trend = Reporter(store).trends(metric_id="builtin/exact_match")

    assert [point.model_dump() for point in trend.points] == [
        {
            "run_id": run_id,
            "metric_id": "builtin/exact_match",
            "value": 1.0,
            "baseline_label": None,
            "created_at": baseline.created_at,
        },
        {
            "run_id": candidate_run_id,
            "metric_id": "builtin/exact_match",
            "value": 0.75,
            "baseline_label": "candidate",
            "created_at": store._run_records[candidate_run_id].created_at,
        },
    ]


def test_reporter_flags_threshold_regressions_against_baseline_label() -> None:
    store, run_id = _store()
    baseline = store.get_run_record(run_id)
    assert baseline is not None
    store.update_run_record(run_id, baseline_label="main")
    candidate_run_id = "candidate-run"
    store._run_records[candidate_run_id] = RunRecord(
        run_id=candidate_run_id,
        status="completed",
        dataset_source_ids=["dataset-1"],
        dataset_fingerprints=["fingerprint-1"],
        metric_ids=["builtin/exact_match"],
    )
    store._projections[(candidate_run_id, "benchmark_result")] = {
        "run_id": candidate_run_id,
        "dataset_ids": ["dataset-1"],
        "metric_ids": ["builtin/exact_match"],
        "total_cases": 1,
        "completed_cases": 1,
        "failed_cases": 0,
        "score_rows": [],
        "metric_means": {"builtin/exact_match": 0.8},
        "outcome_counts": {},
        "error_counts": {},
    }

    summary = Reporter(store).regressions(
        candidate_run_id,
        RegressionPolicy(
            baseline_label="main",
            metric_thresholds={"builtin/exact_match": -0.1},
        ),
    )

    assert [finding.model_dump() for finding in summary.findings] == [
        {
            "metric_id": "builtin/exact_match",
            "baseline_run_id": run_id,
            "candidate_run_id": candidate_run_id,
            "baseline_value": 1.0,
            "candidate_value": 0.8,
            "delta": -0.2,
            "threshold": -0.1,
            "regressed": True,
        }
    ]


def test_snapshot_report_includes_identity_and_provenance() -> None:
    snapshot = _snapshot()

    report = snapshot_report(snapshot, {"stored_events": 6})
    identity = cast(dict[str, JSONValue], report["identity"])
    dataset_source_refs = cast(list[JSONValue], identity["dataset_source_refs"])
    first_dataset_ref = cast(dict[str, JSONValue], dataset_source_refs[0])
    provenance = cast(dict[str, JSONValue], report["provenance"])

    assert report["run_id"] == snapshot.run_id
    assert first_dataset_ref["dataset_id"] == "dataset-1"
    assert provenance["themis_version"] == CURRENT_VERSION
    assert report["run_metadata"] == {"stored_events": 6}


def test_quickcheck_summarizes_completed_run_from_store() -> None:
    store, run_id = _store()

    summary = quickcheck(store, run_id)

    assert summary["run_id"] == run_id
    assert summary["status"] == "completed"
    assert summary["total_cases"] == 1
    assert summary["completed_cases"] == 1
    assert summary["failed_cases"] == 0
    assert summary["metric_means"] == {"builtin/exact_match": 1.0}
