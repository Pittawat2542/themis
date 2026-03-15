from __future__ import annotations

import pytest

from themis.orchestration.projection_handler import ProjectionHandler
from themis.orchestration.task_resolution import resolve_task_stages
from themis.records.trial import TrialRecord
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
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage._protocols import StorageConnectionManager
from themis.types.enums import RecordStatus, DatasetSource
from themis.types.events import (
    ProjectionCompletedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventMetadata,
    TrialEventType,
)
from typing import Any, cast


def _make_trial() -> TrialSpec:
    return TrialSpec(
        trial_id="trial_projection_handler",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
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
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )


def test_projection_handler_appends_projection_event_and_materializes_overlay(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_handler.db")
    manager.initialize()

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager)
    )
    handler = ProjectionHandler(
        event_repo=event_repo,
        projection_repo=projection_repo,
        projection_version="test-v2",
    )

    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    event_repo.save_spec(trial)

    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.ITEM_LOADED,
            stage=TimelineStage.ITEM_LOAD,
            metadata=cast(
                TrialEventMetadata,
                {"item_id": trial.item_id, "dataset_source": "memory"},
            ),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.EVALUATION_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.EVALUATION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash,
                    "metric_id": "exact_match",
                    "score": 1.0,
                },
            ),
            payload={
                "spec_hash": evaluation_hash,
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)

    record = handler.on_trial_completed(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )

    assert record.spec_hash == trial.spec_hash
    projection_event = event_repo.get_events(trial.spec_hash)[-1]
    assert projection_event.event_type == "projection_completed"
    assert isinstance(projection_event.metadata, ProjectionCompletedEventMetadata)
    assert projection_event.metadata.transform_hash == transform_hash
    assert projection_event.metadata.evaluation_hash == evaluation_hash
    assert (
        projection_repo.get_record_timeline(
            trial.spec_hash,
            "trial",
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
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
                    event_type=TrialEventType.TRIAL_COMPLETED,
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
        def materialize_trial_record(
            self,
            trial_hash,
            *,
            transform_hash=None,
            evaluation_hash=None,
            extra_events=None,
            conn=None,
        ):
            raise RuntimeError("projection failed")

    event_repo = InMemoryEventRepo()
    handler = ProjectionHandler(
        event_repo=cast(Any, event_repo),
        projection_repo=cast(Any, FailingProjectionRepo()),
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        handler.on_trial_completed(
            "trial_hash",
            transform_hash="tx_hash",
            evaluation_hash="eval_hash",
        )

    assert [event.event_type for event in event_repo.events] == ["trial_completed"]


def test_projection_handler_uses_explicit_refresh_policy() -> None:
    class InMemoryEventRepo:
        def get_events(self, trial_hash):
            del trial_hash
            return []

        def last_event_index(self, trial_hash):
            del trial_hash
            return None

        def append_event(self, event):
            del event

    class RecordingProjectionRepo:
        def materialize_trial_record(
            self,
            trial_hash,
            *,
            transform_hash=None,
            evaluation_hash=None,
            extra_events=None,
            conn=None,
        ):
            del trial_hash, transform_hash, evaluation_hash, extra_events, conn
            return TrialRecord(
                spec_hash="trial_hash",
                status=RecordStatus.OK,
                candidates=[],
            )

    handler = ProjectionHandler(
        event_repo=cast(Any, InMemoryEventRepo()),
        projection_repo=cast(Any, RecordingProjectionRepo()),
    )

    assert hasattr(handler, "refresh_policy")
    assert not hasattr(handler, "_event_relevant_to_overlay")


def test_projection_handler_refreshes_overlay_when_newer_events_exist(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_refresh.db")
    manager.initialize()

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager)
    )
    handler = ProjectionHandler(
        event_repo=event_repo,
        projection_repo=projection_repo,
        projection_version="test-v2",
    )

    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    event_repo.save_spec(trial)

    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type=TrialEventType.EVALUATION_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.EVALUATION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash,
                    "metric_id": "exact_match",
                    "score": 1.0,
                },
            ),
            payload={
                "spec_hash": evaluation_hash,
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)

    first_record = handler.on_trial_completed(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
    assert first_record.candidates[0].evaluation is not None
    assert first_record.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0

    event_repo.append_event(
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type=TrialEventType.EVALUATION_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.EVALUATION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash,
                    "metric_id": "exact_match",
                    "score": 0.0,
                },
            ),
            payload={
                "spec_hash": evaluation_hash,
                "metric_scores": [{"metric_id": "exact_match", "value": 0.0}],
            },
        )
    )

    refreshed_record = handler.on_trial_completed(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
    projection_events = [
        event
        for event in event_repo.get_events(trial.spec_hash)
        if event.event_type == TrialEventType.PROJECTION_COMPLETED
    ]

    assert len(projection_events) == 2
    assert refreshed_record.candidates[0].evaluation is not None
    assert (
        refreshed_record.candidates[0].evaluation.aggregate_scores["exact_match"] == 0.0
    )


def test_projection_handler_ignores_newer_events_for_other_overlays() -> None:
    class InMemoryEventRepo:
        def __init__(self) -> None:
            self.events = [
                TrialEvent(
                    trial_hash="trial_hash",
                    event_seq=1,
                    event_id="evt_1",
                    event_type=TrialEventType.TRIAL_COMPLETED,
                    payload={"status": "ok"},
                ),
                TrialEvent(
                    trial_hash="trial_hash",
                    event_seq=2,
                    event_id="evt_2",
                    event_type=TrialEventType.PROJECTION_COMPLETED,
                    stage=TimelineStage.PROJECTION,
                    metadata=cast(
                        TrialEventMetadata,
                        {
                            "transform_hash": "tx_a",
                            "projection_version": "v2",
                        },
                    ),
                ),
                TrialEvent(
                    trial_hash="trial_hash",
                    event_seq=3,
                    event_id="evt_3",
                    event_type=TrialEventType.EXTRACTION_COMPLETED,
                    candidate_id="candidate_1",
                    stage=TimelineStage.EXTRACTION,
                    metadata=cast(
                        TrialEventMetadata,
                        {
                            "transform_hash": "tx_b",
                            "extractor_id": "other",
                            "success": True,
                        },
                    ),
                ),
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

    class RecordingProjectionRepo:
        def __init__(self) -> None:
            self.materialize_calls: list[dict[str, object]] = []

        def materialize_trial_record(
            self,
            trial_hash,
            *,
            transform_hash=None,
            evaluation_hash=None,
            extra_events=None,
            conn=None,
        ):
            del conn
            self.materialize_calls.append(
                {
                    "trial_hash": trial_hash,
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash,
                    "extra_events": extra_events,
                }
            )
            return TrialRecord(
                spec_hash=trial_hash,
                status=RecordStatus.OK,
                candidates=[],
            )

    event_repo = InMemoryEventRepo()
    projection_repo = RecordingProjectionRepo()
    handler = ProjectionHandler(
        event_repo=cast(Any, event_repo), projection_repo=cast(Any, projection_repo)
    )

    handler.on_trial_completed("trial_hash", transform_hash="tx_a")

    assert projection_repo.materialize_calls == [
        {
            "trial_hash": "trial_hash",
            "transform_hash": "tx_a",
            "evaluation_hash": None,
            "extra_events": None,
        }
    ]
    assert [event.event_type for event in event_repo.events] == [
        TrialEventType.TRIAL_COMPLETED,
        TrialEventType.PROJECTION_COMPLETED,
        TrialEventType.EXTRACTION_COMPLETED,
    ]
