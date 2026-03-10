import pytest

from themis.errors.exceptions import StorageError
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage.event_repo import SqliteEventRepository
from themis.contracts.protocols import TrialEventRepository
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec
from themis.storage.events import ArtifactRef, TrialEvent
from themis.types.events import TrialEventType


def test_sqlite_event_repo(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    manager.initialize()

    repo = SqliteEventRepository(manager)
    assert isinstance(repo, TrialEventRepository)

    # Needs a spec to satisfy foreign key constraints before appending events
    spec = TrialSpec(
        trial_id="trial_event_repo",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    repo.save_spec(spec)

    event = TrialEvent(
        trial_hash=spec.spec_hash,
        event_seq=1,
        event_id="evt_1",
        event_type="inference_completed",
        candidate_id="candidate_1",
        stage="inference",
        metadata={"provider": "fake"},
        payload={"time": "now"},
        artifact_refs=[
            ArtifactRef(
                artifact_hash="sha256:deadbeef",
                media_type="application/json",
                label="inference",
            )
        ],
    )
    repo.append_event(event)

    events = repo.get_events(spec.spec_hash)
    assert len(events) == 1
    assert events[0] == event
    assert repo.last_event_index(spec.spec_hash) == 1
    assert repo.last_event_index(spec.spec_hash, candidate_id="candidate_1") == 1


def test_sqlite_event_repo_wraps_hydration_validation_errors(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/invalid_event.db")
    manager.initialize()
    repo = SqliteEventRepository(manager)

    spec = TrialSpec(
        trial_id="trial_event_repo",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    repo.save_spec(spec)

    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO trial_events (
                    trial_hash,
                    event_seq,
                    event_id,
                    candidate_id,
                    event_type,
                    stage,
                    status,
                    event_ts,
                    metadata_json,
                    payload_json,
                    artifact_refs_json,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.spec_hash,
                    1,
                    "evt_invalid",
                    "candidate_1",
                    "inference_completed",
                    "inference",
                    None,
                    "2026-03-10T00:00:00+00:00",
                    "{}",
                    "{}",
                    '[{"media_type": "application/json"}]',
                    None,
                ),
            )

    with pytest.raises(StorageError, match="trial_events.artifact_refs_json"):
        repo.get_events(spec.spec_hash)


def test_latest_terminal_event_type_ignores_malformed_non_terminal_payloads(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/terminal_lookup.db")
    manager.initialize()
    repo = SqliteEventRepository(manager)

    spec = TrialSpec(
        trial_id="trial_event_repo",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    repo.save_spec(spec)

    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO trial_events (
                    trial_hash,
                    event_seq,
                    event_id,
                    candidate_id,
                    event_type,
                    stage,
                    status,
                    event_ts,
                    metadata_json,
                    payload_json,
                    artifact_refs_json,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.spec_hash,
                    1,
                    "evt_invalid_non_terminal",
                    "candidate_1",
                    "inference_completed",
                    "inference",
                    None,
                    "2026-03-10T00:00:00+00:00",
                    "{}",
                    "{not-json",
                    "[]",
                    None,
                ),
            )
            conn.execute(
                """
                INSERT INTO trial_events (
                    trial_hash,
                    event_seq,
                    event_id,
                    candidate_id,
                    event_type,
                    stage,
                    status,
                    event_ts,
                    metadata_json,
                    payload_json,
                    artifact_refs_json,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.spec_hash,
                    2,
                    "evt_terminal",
                    None,
                    "trial_completed",
                    None,
                    "ok",
                    "2026-03-10T00:00:01+00:00",
                    "{}",
                    '{"status":"ok"}',
                    "[]",
                    None,
                ),
            )

    assert (
        repo.latest_terminal_event_type(spec.spec_hash)
        == TrialEventType.TRIAL_COMPLETED
    )


def test_has_projection_for_revision_ignores_malformed_unrelated_events(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_lookup.db")
    manager.initialize()
    repo = SqliteEventRepository(manager)

    spec = TrialSpec(
        trial_id="trial_event_repo",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    repo.save_spec(spec)

    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO trial_events (
                    trial_hash,
                    event_seq,
                    event_id,
                    candidate_id,
                    event_type,
                    stage,
                    status,
                    event_ts,
                    metadata_json,
                    payload_json,
                    artifact_refs_json,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.spec_hash,
                    1,
                    "evt_invalid_unrelated",
                    "candidate_1",
                    "inference_completed",
                    "inference",
                    None,
                    "2026-03-10T00:00:00+00:00",
                    "{}",
                    "{}",
                    "[not-json",
                    None,
                ),
            )
            conn.execute(
                """
                INSERT INTO trial_events (
                    trial_hash,
                    event_seq,
                    event_id,
                    candidate_id,
                    event_type,
                    stage,
                    status,
                    event_ts,
                    metadata_json,
                    payload_json,
                    artifact_refs_json,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.spec_hash,
                    2,
                    "evt_projection",
                    None,
                    "projection_completed",
                    "projection",
                    "ok",
                    "2026-03-10T00:00:01+00:00",
                    '{"eval_revision": "latest", "projection_version": "v1"}',
                    "{}",
                    "[]",
                    None,
                ),
            )

    assert repo.has_projection_for_revision(spec.spec_hash, "latest") is True
