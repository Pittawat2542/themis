from __future__ import annotations

import json
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import jsonl_store


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
        storage=StorageConfig(target="jsonl"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        seeds=[7],
    )
    return experiment.compile()


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

    assert [event.event_type for event in store.query_events(snapshot.run_id)] == [
        "run_started"
    ]
