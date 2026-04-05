from __future__ import annotations

import json

from themis.core.base import JSONValue
from themis.core.bundles import (
    export_parse_bundle,
    export_reduction_bundle,
    export_score_bundle,
    export_evaluation_bundle,
    export_generation_bundle,
    import_parse_bundle,
    import_reduction_bundle,
    import_score_bundle,
    import_evaluation_bundle,
    import_generation_bundle,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    ScoreCompletedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores.memory import InMemoryRunStore


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
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
            result={
                "candidate_id": "case-1-candidate-7",
                "final_output": {"answer": "4"},
            },
        )
    )
    store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-11",
            candidate_index=1,
            seed=11,
            result={
                "candidate_id": "case-1-candidate-11",
                "final_output": {"answer": "4"},
            },
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
            result={
                "candidate_id": "case-1-candidate-7",
                "final_output": {"answer": "4"},
            },
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


def test_import_evaluation_bundle_preserves_partial_failures() -> None:
    snapshot = _snapshot()
    source_store = InMemoryRunStore()
    source_store.initialize()
    source_store.persist_snapshot(snapshot)
    execution_payload: dict[str, JSONValue] = {
        "execution_id": "execution-1",
        "subject_kind": "candidate_set",
        "scores": [{"metric_id": "metric/judge", "value": 1.0}],
        "failures": [
            {
                "call_id": "call-2",
                "step_id": "call-2:model_call",
                "step_type": "model_call",
                "error_message": "judge timeout",
            }
        ],
        "status": "partial_failure",
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
    execution = resumed.execution_state.case_states["case-1"].evaluation_executions[
        "metric/judge"
    ]
    assert execution.status == "partial_failure"
    assert execution.failures[0].error_message == "judge timeout"


def test_reduction_parse_and_score_bundles_round_trip_stage_artifacts() -> None:
    snapshot = _snapshot()
    source_store = InMemoryRunStore()
    source_store.initialize()
    source_store.persist_snapshot(snapshot)
    source_store.persist_event(
        ReductionCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            source_candidate_ids=["case-1-candidate-7", "case-1-candidate-11"],
            result={
                "candidate_id": "case-1-reduced",
                "source_candidate_ids": ["case-1-candidate-7", "case-1-candidate-11"],
                "final_output": {"answer": "4"},
            },
        )
    )
    source_store.persist_event(
        ParseCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            result={"value": {"answer": "4"}, "format": "json"},
        )
    )
    source_store.persist_event(
        ScoreCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="builtin/exact_match",
            score={"metric_id": "builtin/exact_match", "value": 1.0},
        )
    )

    reduction_bundle = export_reduction_bundle(source_store, snapshot.run_id)
    parse_bundle = export_parse_bundle(source_store, snapshot.run_id)
    score_bundle = export_score_bundle(source_store, snapshot.run_id)

    target_store = InMemoryRunStore()
    target_store.initialize()
    import_reduction_bundle(target_store, reduction_bundle)
    import_parse_bundle(target_store, parse_bundle)
    import_score_bundle(target_store, score_bundle)

    resumed = target_store.resume(snapshot.run_id)

    assert resumed is not None
    assert [event.event_type for event in resumed.events] == [
        "reduction_completed",
        "parse_completed",
        "score_completed",
    ]
    case_state = resumed.execution_state.case_states["case-1"]
    assert case_state.reduced_candidate is not None
    assert case_state.parsed_output is not None
    assert case_state.successful_scores["builtin/exact_match"].value == 1.0
