import pytest

from themis.errors.exceptions import StorageError
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.event_repo import SqliteEventRepository
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec
from themis.storage.events import TrialEvent
from themis.types.enums import RecordStatus


def test_projection_repo_materializes_trial_record_from_event_log(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_test.db")
    manager.initialize()

    repo = SqliteProjectionRepository(manager)
    event_repo = SqliteEventRepository(manager)

    trial = TrialSpec(
        trial_id="trial_projection",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    event_repo.save_spec(trial)
    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type="item_loaded",
            stage="item_load",
            metadata={"item_id": trial.item_id, "dataset_source": "memory"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type="prompt_rendered",
            stage="prompt_render",
            metadata={"prompt_template_id": "baseline"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type="candidate_started",
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type="inference_completed",
            candidate_id="candidate_1",
            stage="inference",
            metadata={"provider": "fake", "model_id": "test"},
            payload={"spec_hash": "inf_hash", "raw_text": "42"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type="evaluation_completed",
            candidate_id="candidate_1",
            stage="evaluation",
            metadata={"metric_id": "exact_match", "score": 1.0},
            payload={
                "spec_hash": "candidate_1",
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type="candidate_completed",
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id="evt_7",
            event_type="projection_completed",
            stage="projection",
            metadata={"eval_revision": "latest", "projection_version": "v1"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=8,
            event_id="evt_8",
            event_type="trial_completed",
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)

    record = repo.materialize_trial_record(trial.spec_hash, "latest")

    assert record.spec_hash == trial.spec_hash
    assert record.trial_spec == trial
    assert record.status == RecordStatus.OK
    assert len(record.candidates) == 1
    assert record.candidates[0].spec_hash == "candidate_1"
    assert record.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0
    assert record.timeline is not None
    assert [stage.stage for stage in record.timeline.stages] == [
        "item_load",
        "prompt_render",
        "projection",
    ]


def test_projection_repo_wraps_invalid_persisted_trial_specs(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_invalid.db")
    manager.initialize()
    repo = SqliteProjectionRepository(manager)

    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO specs (spec_hash, spec_type, schema_version, canonical_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "trial_invalid",
                    "TrialSpec",
                    "1.0",
                    '{"trial_id": 123}',
                ),
            )

    with pytest.raises(StorageError, match="specs.canonical_json"):
        repo.materialize_trial_record("trial_invalid", "latest")
