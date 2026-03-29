from __future__ import annotations

import json

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import create_run_store, jsonl_store


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
        storage=StorageConfig(store="jsonl"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
            )
        ],
        seeds=[7],
    )
    return experiment.compile()


def test_jsonl_store_persists_snapshot_events_projections_and_blobs(tmp_path) -> None:
    root = tmp_path / "jsonl-store"
    store = jsonl_store(root)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    blob_ref = store.store_blob(b'{"answer":"4"}', "application/json")

    run_root = root / "runs" / snapshot.run_id

    assert (run_root / "snapshot.json").is_file()
    assert (run_root / "events.jsonl").is_file()
    assert (run_root / "projections" / "run_result.json").is_file()
    assert store.load_blob(blob_ref) == ("application/json", b'{"answer":"4"}')
    assert store.resume(snapshot.run_id) is not None
    assert [event.event_type for event in store.query_events(snapshot.run_id)] == ["run_started", "run_completed"]

    timeline_view = store.get_projection(snapshot.run_id, "timeline_view")
    assert timeline_view is not None
    assert [entry["event_type"] for entry in timeline_view["entries"]] == ["run_started", "run_completed"]


def test_jsonl_store_skips_unknown_event_types_on_read(tmp_path) -> None:
    root = tmp_path / "jsonl-store"
    store = jsonl_store(root)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    events_path = root / "runs" / snapshot.run_id / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "2",
                    "event_type": "future_event",
                    "run_id": snapshot.run_id,
                }
            )
            + "\n"
        )

    assert [event.event_type for event in store.query_events(snapshot.run_id)] == ["run_started"]


def test_store_factory_can_build_jsonl_backend(tmp_path) -> None:
    store = create_run_store(StorageConfig(store="jsonl", parameters={"root": str(tmp_path / "jsonl-store")}))

    assert store.__class__.__name__ == "JsonlRunStore"
