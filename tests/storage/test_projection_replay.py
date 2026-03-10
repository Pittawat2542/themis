from __future__ import annotations

from pydantic import TypeAdapter

from themis.records.conversation import ConversationEvent, MessageEvent, MessagePayload
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.events import ArtifactRef, ArtifactRole, ScoreRow, TrialEvent
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec


def test_projection_repo_builds_timelines_views_and_score_rows(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_replay.db")
    manager.initialize()

    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager)

    trial = TrialSpec(
        trial_id="trial_timeline",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-7",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    event_repo.save_spec(trial)

    events = [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type="item_loaded",
            stage="item_load",
            metadata={
                "item_id": trial.item_id,
                "dataset_source": "memory",
                "item_payload_hash": "sha256:item",
            },
            payload={"question": "What is 6 * 7?"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type="prompt_rendered",
            stage="prompt_render",
            metadata={
                "prompt_template_id": "baseline",
                "rendered_prompt_hash": "sha256:prompt",
                "input_field_map": ["question"],
            },
            payload={"messages": [{"role": "user", "content": "Solve the problem."}]},
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
            event_type="prompt_rendered",
            candidate_id="candidate_1",
            stage="prompt_render",
            metadata={
                "prompt_template_id": "baseline",
                "rendered_prompt_hash": "sha256:prompt",
                "input_field_map": ["question"],
            },
            payload={"messages": [{"role": "user", "content": "Solve the problem."}]},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type="conversation_event",
            candidate_id="candidate_1",
            payload=MessageEvent(
                role="assistant",
                payload=MessagePayload(content="The answer is 42."),
                event_index=0,
            ).model_dump(mode="json"),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type="inference_completed",
            candidate_id="candidate_1",
            stage="inference",
            metadata={"provider": "openai", "model_id": "gpt-4o-mini"},
            payload={"spec_hash": "inf_hash", "raw_text": "The answer is 42."},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id="evt_7",
            event_type="extraction_completed",
            candidate_id="candidate_1",
            stage="extraction",
            metadata={"extractor_id": "first_number", "success": True},
            payload={
                "spec_hash": "ext_hash",
                "extractor_id": "first_number",
                "success": True,
                "parsed_answer": "42",
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=8,
            event_id="evt_8",
            event_type="evaluation_completed",
            candidate_id="candidate_1",
            stage="evaluation",
            metadata={"metric_id": "exact_match", "score": 1.0, "judge_call_count": 0},
            payload={
                "spec_hash": "candidate_1",
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=9,
            event_id="evt_9",
            event_type="candidate_completed",
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=10,
            event_id="evt_10",
            event_type="projection_completed",
            stage="projection",
            metadata={"eval_revision": "latest", "projection_version": "v1"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=11,
            event_id="evt_11",
            event_type="trial_completed",
            payload={"status": "ok"},
        ),
    ]
    for event in events:
        event_repo.append_event(event)

    projection_repo.materialize_trial_record(trial.spec_hash, "latest")

    candidate_timeline = projection_repo.get_record_timeline(
        "candidate_1", "candidate", "latest"
    )
    assert candidate_timeline is not None
    assert [stage.stage for stage in candidate_timeline.stages] == [
        "prompt_render",
        "inference",
        "extraction",
        "evaluation",
    ]

    timeline_view = projection_repo.get_timeline_view(
        "candidate_1", "candidate", "latest"
    )
    assert timeline_view is not None
    assert timeline_view.item_payload == {"question": "What is 6 * 7?"}
    assert timeline_view.inference is not None
    assert timeline_view.inference.raw_text == "The answer is 42."
    assert timeline_view.extractions[0].parsed_answer == "42"
    assert timeline_view.evaluation is not None
    assert timeline_view.evaluation.aggregate_scores["exact_match"] == 1.0
    assert timeline_view.conversation is not None
    adapter = TypeAdapter(ConversationEvent)
    conversation_event = next(
        event
        for event in timeline_view.related_events
        if event.event_type == "conversation_event"
    )
    assert (
        adapter.validate_python(conversation_event.payload).payload.content
        == "The answer is 42."
    )
    trial_record = projection_repo.get_trial_record(trial.spec_hash, "latest")
    assert trial_record is not None
    candidate = trial_record.candidates[0]
    assert candidate.candidate_id == "candidate_1"
    assert candidate.sample_index == 0
    assert candidate.conversation is not None
    assert candidate.timeline is not None

    rows = list(
        projection_repo.iter_candidate_scores(
            trial_hash=trial.spec_hash,
            metric_id="exact_match",
            eval_revision="latest",
        )
    )
    assert rows == [
        ScoreRow(
            trial_hash=trial.spec_hash,
            candidate_id="candidate_1",
            metric_id="exact_match",
            score=1.0,
        )
    ]


def test_projection_repo_hydrates_item_payload_from_blob_refs(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_hydrate.db")
    manager.initialize()
    artifact_store = ArtifactStore(base_path=tmp_path / "artifacts", manager=manager)

    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager, artifact_store=artifact_store)

    trial = TrialSpec(
        trial_id="trial_blob_payload",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-blob",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    event_repo.save_spec(trial)
    item_payload = b'{"question":"What is 6 * 7?"}'
    blob_ref = artifact_store.put_blob(item_payload, "application/json")

    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type="item_loaded",
            stage="item_load",
            metadata={
                "item_id": trial.item_id,
                "dataset_source": "memory",
                "item_payload_hash": blob_ref,
            },
            payload=None,
            artifact_refs=[
                ArtifactRef(
                    artifact_hash=blob_ref,
                    media_type="application/json",
                    label="item_payload",
                    role=ArtifactRole.ITEM_PAYLOAD,
                )
            ],
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type="projection_completed",
            stage="projection",
            metadata={"eval_revision": "latest", "projection_version": "v1"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type="trial_completed",
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)

    projection_repo.materialize_trial_record(trial.spec_hash, "latest")
    view = projection_repo.get_timeline_view(trial.spec_hash, "trial", "latest")

    assert view is not None
    assert view.item_payload == {"question": "What is 6 * 7?"}


def test_projection_repo_respects_eval_revision_for_records_and_score_rows(
    tmp_path,
) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_revisions.db")
    manager.initialize()

    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager)

    trial = TrialSpec(
        trial_id="trial_revisions",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source="memory", revision="rev-a"),
            default_metrics=["exact_match"],
        ),
        item_id="item-7",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    event_repo.save_spec(trial)

    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type="candidate_started",
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type="evaluation_completed",
            candidate_id="candidate_1",
            stage="evaluation",
            metadata={"metric_id": "exact_match", "score": 1.0, "judge_call_count": 0},
            payload={
                "spec_hash": "candidate_1",
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type="candidate_completed",
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type="trial_completed",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type="projection_completed",
            stage="projection",
            metadata={"eval_revision": "r1", "projection_version": "v1"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type="evaluation_completed",
            candidate_id="candidate_1",
            stage="evaluation",
            metadata={"metric_id": "exact_match", "score": 0.0, "judge_call_count": 0},
            payload={
                "spec_hash": "candidate_1",
                "metric_scores": [{"metric_id": "exact_match", "value": 0.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id="evt_7",
            event_type="projection_completed",
            stage="projection",
            metadata={"eval_revision": "r2", "projection_version": "v1"},
        ),
    ]:
        event_repo.append_event(event)

    projection_repo.materialize_trial_record(trial.spec_hash, "r1")
    projection_repo.materialize_trial_record(trial.spec_hash, "r2")

    r1 = projection_repo.get_trial_record(trial.spec_hash, "r1")
    r2 = projection_repo.get_trial_record(trial.spec_hash, "r2")
    assert r1 is not None
    assert r2 is not None
    assert r1.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0
    assert r2.candidates[0].evaluation.aggregate_scores["exact_match"] == 0.0

    assert list(
        projection_repo.iter_candidate_scores(
            trial_hash=trial.spec_hash,
            metric_id="exact_match",
            eval_revision="r1",
        )
    ) == [
        ScoreRow(
            trial_hash=trial.spec_hash,
            candidate_id="candidate_1",
            metric_id="exact_match",
            score=1.0,
        )
    ]
    assert list(
        projection_repo.iter_candidate_scores(
            trial_hash=trial.spec_hash,
            metric_id="exact_match",
            eval_revision="r2",
        )
    ) == [
        ScoreRow(
            trial_hash=trial.spec_hash,
            candidate_id="candidate_1",
            metric_id="exact_match",
            score=0.0,
        )
    ]
