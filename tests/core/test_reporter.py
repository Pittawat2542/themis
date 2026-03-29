from __future__ import annotations

import csv
import json
from io import StringIO

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
from themis.core.reporter import Reporter, snapshot_report
from themis.core.stores.memory import InMemoryRunStore


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version="4.0.0a0",
        python_version="3.12.9",
        platform="macos",
    )
    return experiment.compile()


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
            metric_id="metric/demo",
            score={"metric_id": "metric/demo", "value": 1.0, "details": {"matched": True}},
        )
    )
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    return store, snapshot.run_id


def test_reporter_exports_valid_json_markdown_csv_and_latex() -> None:
    store, run_id = _store()
    reporter = Reporter(store)

    exported_json = reporter.export_json(run_id)
    exported_markdown = reporter.export_markdown(run_id)
    exported_csv = reporter.export_csv(run_id)
    exported_latex = reporter.export_latex(run_id)
    score_table = reporter.export_score_table(run_id)

    parsed_json = json.loads(exported_json)
    csv_rows = list(csv.DictReader(StringIO(exported_csv)))

    assert parsed_json["run_result"]["run_id"] == run_id
    assert "# Run Report" in exported_markdown
    assert "metric/demo" in exported_markdown
    assert len(csv_rows) == 1
    assert csv_rows[0]["case_id"] == "case-1"
    assert csv_rows[0]["metric_id"] == "metric/demo"
    assert "\\begin{tabular}" in exported_latex
    assert score_table == [{"case_id": "case-1", "metric_id": "metric/demo", "value": 1.0, "candidate_id": "case-1-reduced"}]


def test_snapshot_report_includes_identity_and_provenance() -> None:
    snapshot = _snapshot()

    report = snapshot_report(snapshot, {"stored_events": 6})

    assert report["run_id"] == snapshot.run_id
    assert report["identity"]["dataset_refs"][0]["dataset_id"] == "dataset-1"
    assert report["provenance"]["themis_version"] == "4.0.0a0"
    assert report["run_metadata"] == {"stored_events": 6}


def test_quickcheck_summarizes_completed_run_from_store() -> None:
    store, run_id = _store()

    summary = quickcheck(store, run_id)

    assert summary["run_id"] == run_id
    assert summary["status"] == "completed"
    assert summary["total_cases"] == 1
    assert summary["completed_cases"] == 1
    assert summary["failed_cases"] == 0
    assert summary["metric_means"] == {"metric/demo": 1.0}
