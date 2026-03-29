from __future__ import annotations

import json

from themis.core.base import JSONValue
from themis.core.bundles import (
    export_evaluation_bundle,
    export_generation_bundle,
    import_evaluation_bundle,
    import_generation_bundle,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import EvaluationCompletedEvent, GenerationCompletedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores.memory import InMemoryRunStore


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 2},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        seeds=[7, 11],
    )
    return experiment.compile()


def test_export_generation_bundle_collects_generation_results_from_store() -> None:
    snapshot = _snapshot()
    store = InMemoryRunStore()
    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-7",
            candidate_index=0,
            seed=7,
            result={"candidate_id": "case-1-candidate-7", "final_output": {"answer": "4"}},
        )
    )
    store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-11",
            candidate_index=1,
            seed=11,
            result={"candidate_id": "case-1-candidate-11", "final_output": {"answer": "4"}},
        )
    )

    bundle = export_generation_bundle(store, snapshot.run_id)

    assert bundle.run_id == snapshot.run_id
    assert bundle.snapshot == snapshot
    assert [record.candidate_id for record in bundle.records] == [
        "case-1-candidate-7",
        "case-1-candidate-11",
    ]


def test_import_generation_bundle_round_trips_generation_events() -> None:
    snapshot = _snapshot()
    source_store = InMemoryRunStore()
    source_store.initialize()
    source_store.persist_snapshot(snapshot)
    source_store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-7",
            candidate_index=0,
            seed=7,
            result={"candidate_id": "case-1-candidate-7", "final_output": {"answer": "4"}},
        )
    )
    bundle = export_generation_bundle(source_store, snapshot.run_id)

    target_store = InMemoryRunStore()
    target_store.initialize()
    import_generation_bundle(target_store, bundle)

    resumed = target_store.resume(snapshot.run_id)

    assert resumed is not None
    assert resumed.snapshot == snapshot
    assert [event.event_type for event in resumed.events] == ["generation_completed"]
    assert isinstance(resumed.events[0], GenerationCompletedEvent)
    assert resumed.events[0].result_blob_ref is not None


def test_export_evaluation_bundle_collects_evaluation_executions_from_store() -> None:
    snapshot = _snapshot()
    store = InMemoryRunStore()
    store.initialize()
    store.persist_snapshot(snapshot)
    execution_payload: dict[str, JSONValue] = {
        "execution_id": "execution-1",
        "subject_kind": "candidate_set",
        "scores": [{"metric_id": "metric/judge", "value": 1.0}],
        "trace": {"trace_id": "trace-1", "steps": []},
    }
    store.persist_event(
        EvaluationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/judge",
            execution=execution_payload,
            execution_blob_ref=store.store_blob(
                json.dumps(execution_payload, sort_keys=True).encode("utf-8"),
                "application/json",
            ),
        )
    )

    bundle = export_evaluation_bundle(store, snapshot.run_id)

    assert bundle.run_id == snapshot.run_id
    assert bundle.snapshot == snapshot
    assert bundle.records[0].metric_id == "metric/judge"
    assert bundle.records[0].candidate_id == "case-1-reduced"
    assert bundle.records[0].execution_blob_ref is not None


def test_import_evaluation_bundle_round_trips_evaluation_events() -> None:
    snapshot = _snapshot()
    source_store = InMemoryRunStore()
    source_store.initialize()
    source_store.persist_snapshot(snapshot)
    execution_payload: dict[str, JSONValue] = {
        "execution_id": "execution-1",
        "subject_kind": "candidate_set",
        "scores": [{"metric_id": "metric/judge", "value": 1.0}],
        "trace": {"trace_id": "trace-1", "steps": []},
    }
    source_store.persist_event(
        EvaluationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/judge",
            execution=execution_payload,
            execution_blob_ref=source_store.store_blob(
                json.dumps(execution_payload, sort_keys=True).encode("utf-8"),
                "application/json",
            ),
        )
    )
    bundle = export_evaluation_bundle(source_store, snapshot.run_id)

    target_store = InMemoryRunStore()
    target_store.initialize()
    import_evaluation_bundle(target_store, bundle)

    resumed = target_store.resume(snapshot.run_id)

    assert resumed is not None
    assert resumed.snapshot == snapshot
    assert [event.event_type for event in resumed.events] == [
        "evaluation_completed",
        "score_completed",
    ]
    assert isinstance(resumed.events[0], EvaluationCompletedEvent)
    assert resumed.events[0].execution_blob_ref == bundle.records[0].execution_blob_ref
