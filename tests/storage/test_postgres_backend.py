from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores.postgres import PostgresRunStore, postgres_store

pytestmark = pytest.mark.skipif(
    not os.getenv("THEMIS_TEST_POSTGRES_ADMIN_URL"),
    reason="THEMIS_TEST_POSTGRES_ADMIN_URL is required for Postgres integration tests",
)


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]),
        storage=StorageConfig(store="postgres"),
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


@pytest.fixture
def postgres_store_backend(tmp_path: Path):
    import psycopg

    admin_url = os.environ["THEMIS_TEST_POSTGRES_ADMIN_URL"]
    database_name = f"themis_test_{uuid.uuid4().hex}"
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(f'CREATE DATABASE "{database_name}"')

    parsed = urlsplit(admin_url)
    database_url = _replace_database(parsed, database_name)
    store = postgres_store(database_url, tmp_path / "postgres-blobs")
    assert isinstance(store, PostgresRunStore)
    store.initialize()

    try:
        yield store, database_url
    finally:
        with psycopg.connect(admin_url, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (database_name,),
            )
            connection.execute(f'DROP DATABASE IF EXISTS "{database_name}"')


def test_postgres_store_round_trips_snapshot_events_projections_and_blobs(postgres_store_backend) -> None:
    store, _database_url = postgres_store_backend
    snapshot = _snapshot()

    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    blob_ref = store.store_blob(b'{"answer":"4"}', "application/json")

    assert store.resume(snapshot.run_id) is not None
    assert [event.event_type for event in store.query_events(snapshot.run_id)] == ["run_started", "run_completed"]
    assert store.get_projection(snapshot.run_id, "run_result") is not None
    assert store.load_blob(blob_ref) == ("application/json", b'{"answer":"4"}')


def test_postgres_store_backfills_missing_projection(postgres_store_backend) -> None:
    store, database_url = postgres_store_backend
    snapshot = _snapshot()

    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    import psycopg

    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            DELETE FROM run_projections
            WHERE run_id = %s AND projection_name = %s
            """,
            (snapshot.run_id, "run_result"),
        )
        connection.commit()

    projection = store.get_projection(snapshot.run_id, "run_result")

    assert projection is not None
    assert projection["status"] == "running"


def test_postgres_store_skips_unknown_event_types_on_read(postgres_store_backend) -> None:
    store, database_url = postgres_store_backend
    snapshot = _snapshot()

    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    import psycopg

    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            INSERT INTO run_events (run_id, event_type, event_json)
            VALUES (%s, %s, %s::jsonb)
            """,
            (
                snapshot.run_id,
                "future_event",
                json.dumps(
                    {
                        "schema_version": "2",
                        "event_type": "future_event",
                        "run_id": snapshot.run_id,
                    }
                ),
            ),
        )
        connection.commit()

    events = store.query_events(snapshot.run_id)

    assert [event.event_type for event in events] == ["run_started"]


def _replace_database(parsed: SplitResult, database_name: str) -> str:
    return urlunsplit(parsed._replace(path=f"/{database_name}"))
