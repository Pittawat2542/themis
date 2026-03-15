from __future__ import annotations
from themis.types.enums import PromptRole

from pydantic import TypeAdapter

from themis.orchestration.task_resolution import resolve_task_stages
from themis.records.conversation import ConversationEvent, MessageEvent, MessagePayload
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
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.events import (
    ArtifactRef,
    ArtifactRole,
    ScoreRow,
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
        trial_id="trial_overlay",
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
        item_id="item-7",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )


def test_projection_repo_builds_overlay_views_and_score_rows(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_overlay.db")
    manager.initialize()

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager)
    )
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    event_repo.save_spec(trial)

    events = [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.ITEM_LOADED,
            stage=TimelineStage.ITEM_LOAD,
            metadata=cast(
                TrialEventMetadata,
                {
                    "item_id": trial.item_id,
                    "dataset_source": "memory",
                    "item_payload_hash": "sha256:item",
                },
            ),
            payload={"question": "What is 6 * 7?"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type=TrialEventType.PROMPT_RENDERED,
            stage=TimelineStage.PROMPT_RENDER,
            metadata=cast(
                TrialEventMetadata,
                {
                    "prompt_template_id": "baseline",
                    "rendered_prompt_hash": "sha256:prompt",
                    "input_field_map": ["question"],
                },
            ),
            payload={"messages": [{"role": "user", "content": "Solve the problem."}]},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type=TrialEventType.PROMPT_RENDERED,
            candidate_id="candidate_1",
            stage=TimelineStage.PROMPT_RENDER,
            metadata=cast(
                TrialEventMetadata,
                {
                    "prompt_template_id": "baseline",
                    "rendered_prompt_hash": "sha256:prompt",
                    "input_field_map": ["question"],
                },
            ),
            payload={"messages": [{"role": "user", "content": "Solve the problem."}]},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type=TrialEventType.CONVERSATION_EVENT,
            candidate_id="candidate_1",
            payload=MessageEvent(
                role=PromptRole.ASSISTANT,
                payload=MessagePayload(content="The answer is 42."),
                event_index=0,
            ).model_dump(mode="json"),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type=TrialEventType.INFERENCE_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.INFERENCE,
            metadata=cast(
                TrialEventMetadata, {"provider": "openai", "model_id": "gpt-4o-mini"}
            ),
            payload={"spec_hash": "inf_hash", "raw_text": "The answer is 42."},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id="evt_7",
            event_type=TrialEventType.EXTRACTION_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.EXTRACTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "extractor_id": "first_number",
                    "success": True,
                },
            ),
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
                    "judge_call_count": 0,
                },
            ),
            payload={
                "spec_hash": evaluation_hash,
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=9,
            event_id="evt_9",
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=10,
            event_id="evt_10",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=11,
            event_id="evt_11",
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
    ]
    for event in events:
        event_repo.append_event(event)

    projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )

    candidate_timeline = projection_repo.get_record_timeline(
        "candidate_1",
        "candidate",
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
    assert candidate_timeline is not None
    assert [stage.stage for stage in candidate_timeline.stages] == [
        "prompt_render",
        "inference",
        "extraction",
        "evaluation",
    ]

    timeline_view = projection_repo.get_timeline_view(
        "candidate_1",
        "candidate",
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
    assert timeline_view is not None
    assert timeline_view.item_payload == {"question": "What is 6 * 7?"}
    assert timeline_view.inference is not None
    assert timeline_view.inference.raw_text == "The answer is 42."
    assert timeline_view.extractions[0].parsed_answer == "42"
    assert timeline_view.evaluation is not None
    assert timeline_view.evaluation.aggregate_scores["exact_match"] == 1.0
    assert timeline_view.conversation is not None
    adapter: TypeAdapter[ConversationEvent] = TypeAdapter(ConversationEvent)
    conversation_event = next(
        event
        for event in timeline_view.related_events
        if event.event_type == "conversation_event"
    )
    parsed_event = adapter.validate_python(conversation_event.payload)
    assert isinstance(parsed_event.payload, MessagePayload)
    assert parsed_event.payload.content == "The answer is 42."
    trial_record = projection_repo.get_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
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
            evaluation_hash=evaluation_hash,
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
    artifact_store = ArtifactStore(
        base_path=tmp_path / "artifacts",
        manager=cast(StorageConnectionManager, manager),
    )

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager), artifact_store=artifact_store
    )

    trial = _make_trial()
    event_repo.save_spec(trial)
    item_payload = b'{"question":"What is 6 * 7?"}'
    blob_ref = artifact_store.put_blob(item_payload, "application/json")

    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.ITEM_LOADED,
            stage=TimelineStage.ITEM_LOAD,
            metadata=cast(
                TrialEventMetadata,
                {
                    "item_id": trial.item_id,
                    "dataset_source": "memory",
                    "item_payload_hash": blob_ref,
                },
            ),
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
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(TrialEventMetadata, {"projection_version": "v2"}),
        ),
    ]:
        event_repo.append_event(event)

    projection_repo.materialize_trial_record(trial.spec_hash)
    view = projection_repo.get_timeline_view(trial.spec_hash, "trial")

    assert view is not None
    assert view.item_payload == {"question": "What is 6 * 7?"}


def test_projection_repo_materializes_same_candidate_for_multiple_evaluations(
    tmp_path,
) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_multiple_eval.db")
    manager.initialize()

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager)
    )
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash_a = resolved.evaluations[0].evaluation_hash
    evaluation_hash_b = "eval_alt_hash"
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
                    "evaluation_hash": evaluation_hash_a,
                    "metric_id": "exact_match",
                    "score": 1.0,
                },
            ),
            payload={
                "spec_hash": evaluation_hash_a,
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
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash_a,
                    "projection_version": "v2",
                },
            ),
        ),
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
                    "evaluation_hash": evaluation_hash_b,
                    "metric_id": "exact_match",
                    "score": 0.0,
                },
            ),
            payload={
                "spec_hash": evaluation_hash_b,
                "metric_scores": [{"metric_id": "exact_match", "value": 0.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id="evt_7",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "evaluation_hash": evaluation_hash_b,
                    "projection_version": "v2",
                },
            ),
        ),
    ]:
        event_repo.append_event(event)

    projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash_a,
    )
    projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash_b,
    )

    record_a = projection_repo.get_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash_a,
    )
    record_b = projection_repo.get_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash_b,
    )
    assert record_a is not None
    assert record_b is not None
    assert record_a.candidates[0].evaluation is not None
    assert record_b.candidates[0].evaluation is not None
    assert record_a.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0
    assert record_b.candidates[0].evaluation.aggregate_scores["exact_match"] == 0.0


def test_projection_repo_applies_overlay_visibility_exactly(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_visibility.db")
    manager.initialize()

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager)
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
            event_type=TrialEventType.INFERENCE_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.INFERENCE,
            metadata=cast(
                TrialEventMetadata, {"provider": "openai", "model_id": "gpt-4o-mini"}
            ),
            payload={"spec_hash": "inf_hash", "raw_text": "The answer is 42."},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.EXTRACTION_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.EXTRACTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "extractor_id": "first_number",
                    "success": True,
                },
            ),
            payload={
                "spec_hash": "ext_hash",
                "extractor_id": "first_number",
                "success": True,
                "parsed_answer": "42",
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
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
            event_seq=5,
            event_id="evt_5",
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)

    generation_record = projection_repo.materialize_trial_record(trial.spec_hash)
    transform_record = projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
    )
    evaluation_record = projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )

    assert generation_record.candidates[0].inference is not None
    assert generation_record.candidates[0].extractions == []
    assert generation_record.candidates[0].evaluation is None
    assert transform_record.candidates[0].extractions[0].parsed_answer == "42"
    assert transform_record.candidates[0].evaluation is None
    assert evaluation_record.candidates[0].extractions[0].parsed_answer == "42"
    assert evaluation_record.candidates[0].evaluation is not None
    assert (
        evaluation_record.candidates[0].evaluation.aggregate_scores["exact_match"]
        == 1.0
    )


def test_projection_repo_materializes_transform_only_overlay(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/projection_transform_only.db")
    manager.initialize()

    event_repo = SqliteEventRepository(cast(StorageConnectionManager, manager))
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, manager)
    )
    trial = _make_trial()
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
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
            event_type=TrialEventType.INFERENCE_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.INFERENCE,
            metadata=cast(
                TrialEventMetadata, {"provider": "openai", "model_id": "gpt-4o-mini"}
            ),
            payload={"spec_hash": "inf_hash", "raw_text": "The answer is 42."},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.EXTRACTION_COMPLETED,
            candidate_id="candidate_1",
            stage=TimelineStage.EXTRACTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash,
                    "extractor_id": "first_number",
                    "success": True,
                },
            ),
            payload={
                "spec_hash": "ext_hash",
                "extractor_id": "first_number",
                "success": True,
                "parsed_answer": "42",
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
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type=TrialEventType.PROJECTION_COMPLETED,
            stage=TimelineStage.PROJECTION,
            metadata=cast(
                TrialEventMetadata,
                {"transform_hash": transform_hash, "projection_version": "v2"},
            ),
        ),
    ]:
        event_repo.append_event(event)

    record = projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
    )

    assert record.candidates[0].extractions[0].parsed_answer == "42"
    assert record.candidates[0].evaluation is None
