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
from themis.core.reporter import Reporter, snapshot_report
from themis.core.stores.memory import InMemoryRunStore


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
        themis_version="4.0.0rc1",
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
            metric_id="builtin/exact_match",
            score={"metric_id": "builtin/exact_match", "value": 1.0, "details": {"matched": True}},
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
    assert parsed_json["snapshot"]["run_id"] == run_id
    assert parsed_json["execution_state"]["run_id"] == run_id
    assert "# Run Report" in exported_markdown
    assert "builtin/exact_match" in exported_markdown
    assert len(csv_rows) == 1
    assert csv_rows[0]["case_id"] == "case-1"
    assert csv_rows[0]["metric_id"] == "builtin/exact_match"
    assert "\\begin{tabular}" in exported_latex
    assert score_table == [{"case_id": "case-1", "metric_id": "builtin/exact_match", "value": 1.0, "candidate_id": "case-1-reduced"}]


def test_reporter_escapes_latex_special_characters() -> None:
    store, run_id = _store()
    store._projections[(run_id, "benchmark_result")] = {
        "run_id": run_id,
        "dataset_ids": ["data_set%1"],
        "metric_ids": ["builtin/exact_match"],
        "total_cases": 1,
        "completed_cases": 1,
        "failed_cases": 0,
        "score_rows": [
            {
                "case_id": r"case_1%&${}\path",
                "metric_id": "metric_^~#",
                "value": r"value_1%&${}\path",
                "candidate_id": None,
            }
        ],
        "metric_means": {"builtin/exact_match": 1.0},
    }
    reporter = Reporter(store)

    exported_latex = reporter.export_latex(run_id)

    assert r"case\_1\%\&\$\{\}\textbackslash{}path" in exported_latex
    assert r"metric\_\textasciicircum{}\textasciitilde{}\#" in exported_latex
    assert r"value\_1\%\&\$\{\}\textbackslash{}path" in exported_latex
    assert " &  \\\\" in exported_latex


def test_snapshot_report_includes_identity_and_provenance() -> None:
    snapshot = _snapshot()

    report = snapshot_report(snapshot, {"stored_events": 6})
    identity = cast(dict[str, JSONValue], report["identity"])
    dataset_refs = cast(list[JSONValue], identity["dataset_refs"])
    first_dataset_ref = cast(dict[str, JSONValue], dataset_refs[0])
    provenance = cast(dict[str, JSONValue], report["provenance"])

    assert report["run_id"] == snapshot.run_id
    assert first_dataset_ref["dataset_id"] == "dataset-1"
    assert provenance["themis_version"] == "4.0.0rc1"
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
