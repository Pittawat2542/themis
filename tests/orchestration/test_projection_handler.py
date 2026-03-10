from __future__ import annotations

import pytest

from themis.orchestration.projection_handler import ProjectionHandler
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.events import TrialEvent
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager


def test_projection_handler_appends_projection_event_and_materializes(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_handler.db")
    manager.initialize()

    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager)
    handler = ProjectionHandler(
        event_repo=event_repo,
        projection_repo=projection_repo,
        projection_version="test-v1",
    )

    trial = TrialSpec(
        trial_id="trial_projection_handler",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
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
            event_seq=5,
            event_id="evt_5",
            event_type="candidate_completed",
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type="trial_completed",
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)

    record = handler.on_trial_completed(trial.spec_hash, eval_revision="latest")

    assert record.spec_hash == trial.spec_hash
    assert [event.event_type for event in event_repo.get_events(trial.spec_hash)][
        -1
    ] == "projection_completed"
    assert (
        projection_repo.get_record_timeline(trial.spec_hash, "trial", "latest")
        is not None
    )


def test_projection_handler_does_not_record_projection_completion_when_materialization_fails():
    class InMemoryEventRepo:
        def __init__(self):
            self.events = [
                TrialEvent(
                    trial_hash="trial_hash",
                    event_seq=1,
                    event_id="evt_1",
                    event_type="trial_completed",
                )
            ]

        def get_events(self, trial_hash):
            return [event for event in self.events if event.trial_hash == trial_hash]

        def last_event_index(self, trial_hash):
            matching = [
                event.event_seq
                for event in self.events
                if event.trial_hash == trial_hash
            ]
            return max(matching) if matching else None

        def append_event(self, event):
            self.events.append(event)

    class FailingProjectionRepo:
        def materialize_trial_record(self, trial_hash, eval_revision):
            raise RuntimeError("projection failed")

    event_repo = InMemoryEventRepo()
    handler = ProjectionHandler(
        event_repo=event_repo, projection_repo=FailingProjectionRepo()
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        handler.on_trial_completed("trial_hash", eval_revision="latest")

    assert [event.event_type for event in event_repo.events] == ["trial_completed"]
