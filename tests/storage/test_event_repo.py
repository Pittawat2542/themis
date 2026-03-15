import json

import pytest
from pydantic import Field

from themis.errors import StorageError
from themis.orchestration.task_resolution import resolve_task_stages
from themis.specs.base import SpecBase
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.events import (
    ArtifactRef,
    EvaluationCompletedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventMetadata,
    TrialEventType,
)
from themis.storage._protocols import StorageConnectionManager
from themis.types.enums import DatasetSource
from typing import cast


def _make_trial() -> TrialSpec:
    return TrialSpec(
        trial_id="trial_event_repo",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="first_number")]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(
                    name="exact_match_eval",
                    transform="json",
                    metrics=["exact_match"],
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )


def test_sqlite_event_repo_round_trips_overlay_metadata(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    manager.initialize()

    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    repo.save_spec(trial)

    event = TrialEvent(
        trial_hash=trial.spec_hash,
        event_seq=1,
        event_id="evt_1",
        event_type=TrialEventType.EVALUATION_COMPLETED,
        candidate_id="candidate_1",
        stage=TimelineStage.EVALUATION,
        metadata=cast(
            TrialEventMetadata,
            {
                "transform_hash": transform_hash,
                "evaluation_hash": evaluation_hash,
                "metric_id": "exact_match",
            },
        ),
        payload={"time": "now"},
        artifact_refs=[
            ArtifactRef(
                artifact_hash="sha256:deadbeef",
                media_type="application/json",
                label="evaluation",
            )
        ],
    )
    repo.append_event(event)

    events = repo.get_events(trial.spec_hash)
    assert len(events) == 1
    assert events[0] == event
    assert isinstance(events[0].metadata, EvaluationCompletedEventMetadata)
    assert events[0].metadata.transform_hash == transform_hash
    assert events[0].metadata.evaluation_hash == evaluation_hash
    assert repo.last_event_index(trial.spec_hash) == 1
    assert repo.last_event_index(trial.spec_hash, candidate_id="candidate_1") == 1


def test_has_projection_for_overlay_matches_exact_overlay_identity(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_identity.db")
    manager.initialize()

    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    repo.save_spec(trial)

    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_generation_projection",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(TrialEventMetadata, {"projection_version": "v2"}),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_transform_projection",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "projection_version": "v2",
                },
            ),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_evaluation_projection",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash,
                    "projection_version": "v2",
                },
            ),
        ),
    ]:
        repo.append_event(event)

    assert repo.has_projection_for_overlay(trial.spec_hash) is True
    assert (
        repo.has_projection_for_overlay(
            trial.spec_hash,
            transform_hash=transform_hash,
        )
        is True
    )
    assert (
        repo.has_projection_for_overlay(
            trial.spec_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        is True
    )
    assert (
        repo.has_projection_for_overlay(
            trial.spec_hash,
            evaluation_hash=evaluation_hash,
        )
        is False
    )
    assert (
        repo.has_projection_for_overlay(
            trial.spec_hash,
            transform_hash="missing",
        )
        is False
    )


def test_save_spec_preserves_stage_only_task_payloads(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/spec_roundtrip.db")
    manager.initialize()

    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    trial = TrialSpec(
        trial_id="trial_stage_only",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            output_transforms=[
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="first_number")]
                    ),
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )

    repo.save_spec(trial)

    with manager.get_connection() as conn:
        row = conn.execute(
            "SELECT canonical_json FROM specs WHERE spec_hash = ?",
            (trial.spec_hash,),
        ).fetchone()

    assert row is not None
    restored = TrialSpec.model_validate(json.loads(row["canonical_json"]))
    assert restored.task.generation is None
    assert [transform.name for transform in restored.task.output_transforms] == ["json"]


def test_save_spec_rejects_short_hash_collision_with_different_identity(
    tmp_path,
) -> None:
    class CollisionSpec(SpecBase):
        name: str
        forced_full_hash: str = Field(exclude=True)

        def compute_hash(self, *, short: bool = False) -> str:
            if short:
                return self.forced_full_hash[:12]
            return self.forced_full_hash

    manager = DatabaseManager(f"sqlite:///{tmp_path}/collision.db")
    manager.initialize()
    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))

    repo.save_spec(
        CollisionSpec(
            name="first",
            forced_full_hash="aaaaaaaaaaaa1111111111111111111111111111111111111111111111111111",
        )
    )

    with pytest.raises(StorageError, match="short hash collision"):
        repo.save_spec(
            CollisionSpec(
                name="second",
                forced_full_hash="aaaaaaaaaaaa2222222222222222222222222222222222222222222222222222",
            )
        )


def test_latest_terminal_event_type_ignores_malformed_non_terminal_payloads(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/terminal_lookup.db")
    manager.initialize()
    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))

    trial = _make_trial()
    repo.save_spec(trial)

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
                    trial.spec_hash,
                    1,
                    "evt_invalid_non_terminal",
                    "candidate_1",
                    "evaluation_completed",
                    "evaluation",
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
                    trial.spec_hash,
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
        repo.latest_terminal_event_type(trial.spec_hash)
        == TrialEventType.TRIAL_COMPLETED
    )


def test_has_projection_for_overlay_ignores_malformed_unrelated_events(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_lookup.db")
    manager.initialize()
    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))

    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    repo.save_spec(trial)

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
                    trial.spec_hash,
                    1,
                    "evt_invalid_unrelated",
                    "candidate_1",
                    "evaluation_completed",
                    "evaluation",
                    None,
                    "2026-03-10T00:00:00+00:00",
                    "{not-json",
                    "{}",
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
                    trial.spec_hash,
                    2,
                    "evt_projection",
                    None,
                    "projection_completed",
                    "projection",
                    "ok",
                    "2026-03-10T00:00:01+00:00",
                    (
                        '{"transform_hash":"%s","evaluation_hash":"%s","projection_version":"v2"}'
                        % (transform_hash, evaluation_hash)
                    ),
                    None,
                    "[]",
                    None,
                ),
            )

    assert (
        repo.has_projection_for_overlay(
            trial.spec_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        is True
    )
    assert (
        repo.has_projection_for_overlay(
            trial.spec_hash,
            transform_hash=transform_hash,
            evaluation_hash="missing",
        )
        is False
    )


def test_sqlite_event_repo_wraps_hydration_validation_errors(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/invalid_event.db")
    manager.initialize()
    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))

    trial = _make_trial()
    repo.save_spec(trial)

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
                    trial.spec_hash,
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
        repo.get_events(trial.spec_hash)


def test_sqlite_event_repo_wraps_structured_metadata_validation_errors(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/invalid_metadata.db")
    manager.initialize()
    repo = SqliteEventRepository(cast(StorageConnectionManager, manager))

    trial = _make_trial()
    repo.save_spec(trial)

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
                    trial.spec_hash,
                    1,
                    "evt_invalid_metadata",
                    None,
                    "projection_completed",
                    "projection",
                    "ok",
                    "2026-03-10T00:00:00+00:00",
                    '{"projection_version":"v2","source_event_range":"bad"}',
                    None,
                    "[]",
                    None,
                ),
            )

    with pytest.raises(StorageError, match="trial_events.metadata_json"):
        repo.get_events(trial.spec_hash)
