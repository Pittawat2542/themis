import pytest

from themis.errors import StorageError
from themis.records.error import ErrorRecord
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.event_repo import SqliteEventRepository
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    GenerationSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.orchestration.task_resolution import resolve_task_stages
from themis.types.events import ScoreRow, TrialSummaryRow
from themis.types.enums import (
    ErrorCode,
    ErrorWhere,
    RecordStatus,
    RecordType,
    DatasetSource,
)
from themis.types.events import (
    TimelineStage,
    TrialEvent,
    TrialEventType,
    TrialEventMetadata,
)
from themis.storage._protocols import StorageConnectionManager
from typing import Any, cast


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
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
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
            event_type="prompt_rendered",
            stage=TimelineStage.PROMPT_RENDER,
            metadata=cast(TrialEventMetadata, {"prompt_template_id": "baseline"}),
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
            stage=TimelineStage.INFERENCE,
            metadata=cast(TrialEventMetadata, {"provider": "fake", "model_id": "test"}),
            payload={"spec_hash": "inf_hash", "raw_text": "42"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type="evaluation_completed",
            candidate_id="candidate_1",
            stage=TimelineStage.EVALUATION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "metric_id": "exact_match",
                    "score": 1.0,
                    "transform_hash": None,
                    "evaluation_hash": "eval_1",
                },
            ),
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
            stage=TimelineStage.PROJECTION,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": None,
                    "evaluation_hash": "eval_1",
                    "projection_version": "v1",
                },
            ),
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

    record = repo.materialize_trial_record(trial.spec_hash, evaluation_hash="eval_1")

    assert record.spec_hash == trial.spec_hash
    assert record.trial_spec == trial
    assert record.status == RecordStatus.OK
    assert len(record.candidates) == 1
    assert record.candidates[0].spec_hash == "candidate_1"
    assert record.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0
    assert record.timeline is not None
    assert record.timeline.record_type == RecordType.TRIAL
    assert [stage.stage for stage in record.timeline.stages] == [
        TimelineStage.ITEM_LOAD,
        TimelineStage.PROMPT_RENDER,
        TimelineStage.PROJECTION,
    ]

    persisted_timeline = repo.get_record_timeline(
        trial.spec_hash,
        "trial",
        evaluation_hash="eval_1",
    )
    assert persisted_timeline is not None
    assert persisted_timeline.record_type == RecordType.TRIAL
    assert persisted_timeline.stages[0].stage == TimelineStage.ITEM_LOAD


def test_projection_repo_delegates_materialization_to_materializer(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_delegate_materializer.db")
    manager.initialize()
    repo = SqliteProjectionRepository(cast(StorageConnectionManager, manager))
    expected = object()

    class StubMaterializer:
        def __init__(self) -> None:
            self.calls: list[
                tuple[str, str | None, str | None, object | None, object | None]
            ] = []

        def materialize_trial_record(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
            extra_events=None,
            conn=None,
        ):
            self.calls.append(
                (trial_hash, transform_hash, evaluation_hash, extra_events, conn)
            )
            return expected

    stub = StubMaterializer()
    repo._materializer = cast(Any, stub)

    result = repo.materialize_trial_record(
        "trial_hash",
        transform_hash="tx_hash",
        evaluation_hash="eval_hash",
    )

    assert result is expected
    assert stub.calls == [("trial_hash", "tx_hash", "eval_hash", None, None)]


def test_projection_repo_get_trial_record_uses_query_service_and_materializer(
    tmp_path,
) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_delegate_record.db")
    manager.initialize()
    repo = SqliteProjectionRepository(cast(StorageConnectionManager, manager))
    expected = object()

    class StubQueries:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None, str | None]] = []

        def overlay_exists_or_materialized(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
            overlay_key: str | None = None,
        ) -> bool:
            del overlay_key
            self.calls.append((trial_hash, transform_hash, evaluation_hash))
            return True

    class StubMaterializer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None, str | None]] = []

        def materialize_trial_record(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
            extra_events=None,
            conn=None,
        ):
            del extra_events, conn
            self.calls.append((trial_hash, transform_hash, evaluation_hash))
            return expected

    queries = StubQueries()
    materializer = StubMaterializer()
    repo._queries = cast(Any, queries)
    repo._materializer = cast(Any, materializer)

    result = repo.get_trial_record(
        "trial_hash",
        transform_hash="tx_hash",
        evaluation_hash="eval_hash",
    )

    assert result is expected
    assert queries.calls == [("trial_hash", "tx_hash", "eval_hash")]
    assert materializer.calls == [("trial_hash", "tx_hash", "eval_hash")]


def test_projection_repo_delegates_query_methods_to_query_service(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_delegate_queries.db")
    manager.initialize()
    repo = SqliteProjectionRepository(cast(StorageConnectionManager, manager))
    expected_score = ScoreRow(
        trial_hash="trial_hash",
        candidate_id="candidate_1",
        metric_id="exact_match",
        score=1.0,
    )
    expected_summary = TrialSummaryRow(
        trial_hash="trial_hash",
        model_id="model",
        task_id="task",
        item_id="item",
        status=RecordStatus.OK,
    )

    class StubQueries:
        def __init__(self) -> None:
            self.score_calls: list[tuple[list[str] | None, str | None, str | None]] = []
            self.summary_calls: list[
                tuple[list[str] | None, str | None, str | None]
            ] = []

        def iter_candidate_scores(
            self,
            *,
            trial_hashes=None,
            metric_id: str | None = None,
            evaluation_hash: str | None = None,
        ):
            self.score_calls.append(
                (
                    list(trial_hashes) if trial_hashes is not None else None,
                    metric_id,
                    evaluation_hash,
                )
            )
            return iter([expected_score])

        def iter_trial_summaries(
            self,
            *,
            trial_hashes=None,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ):
            self.summary_calls.append(
                (
                    list(trial_hashes) if trial_hashes is not None else None,
                    transform_hash,
                    evaluation_hash,
                )
            )
            return iter([expected_summary])

    stub = StubQueries()
    repo._queries = cast(Any, stub)

    score_rows = list(
        repo.iter_candidate_scores(
            trial_hashes=["trial_hash"],
            metric_id="exact_match",
            evaluation_hash="eval_hash",
        )
    )
    summary_rows = list(
        repo.iter_trial_summaries(
            trial_hashes=["trial_hash"],
            transform_hash="tx_hash",
            evaluation_hash="eval_hash",
        )
    )

    assert score_rows == [expected_score]
    assert summary_rows == [expected_summary]
    assert stub.score_calls == [(["trial_hash"], "exact_match", "eval_hash")]
    assert stub.summary_calls == [(["trial_hash"], "tx_hash", "eval_hash")]


def test_projection_repo_delegates_timeline_reads_to_view_service(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_delegate_views.db")
    manager.initialize()
    repo = SqliteProjectionRepository(cast(StorageConnectionManager, manager))
    expected_timeline = object()
    expected_view = object()

    class StubTimelineViews:
        def __init__(self) -> None:
            self.timeline_calls: list[tuple[str, str, str | None, str | None]] = []
            self.view_calls: list[tuple[str, str, str | None, str | None]] = []

        def get_record_timeline(
            self,
            record_id: str,
            record_type,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ):
            self.timeline_calls.append(
                (record_id, str(record_type), transform_hash, evaluation_hash)
            )
            return expected_timeline

        def get_timeline_view(
            self,
            record_id: str,
            record_type,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ):
            self.view_calls.append(
                (record_id, str(record_type), transform_hash, evaluation_hash)
            )
            return expected_view

    stub = StubTimelineViews()
    repo._timeline_views = cast(Any, stub)

    timeline = repo.get_record_timeline(
        "candidate_1",
        RecordType.CANDIDATE,
        transform_hash="tx_hash",
        evaluation_hash="eval_hash",
    )
    view = repo.get_timeline_view(
        "candidate_1",
        RecordType.CANDIDATE,
        transform_hash="tx_hash",
        evaluation_hash="eval_hash",
    )

    assert timeline is expected_timeline
    assert view is expected_view
    assert stub.timeline_calls == [
        ("candidate_1", str(RecordType.CANDIDATE), "tx_hash", "eval_hash")
    ]
    assert stub.view_calls == [
        ("candidate_1", str(RecordType.CANDIDATE), "tx_hash", "eval_hash")
    ]


def test_projection_repo_wraps_invalid_persisted_trial_specs(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_invalid.db")
    manager.initialize()
    repo = SqliteProjectionRepository(manager)

    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO specs (
                    spec_hash,
                    canonical_hash,
                    spec_type,
                    schema_version,
                    canonical_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "trial_invalid",
                    "invalid0000000000000000000000000000000000000000000000000000000000",
                    "TrialSpec",
                    "1.0",
                    '{"trial_id": 123}',
                ),
            )

    with pytest.raises(StorageError, match="specs.canonical_json"):
        repo.materialize_trial_record("trial_invalid")


def test_projection_repo_marks_failed_transform_overlay_as_error_and_not_cached(
    tmp_path,
):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_overlay_error.db")
    manager.initialize()

    repo = SqliteProjectionRepository(manager)
    event_repo = SqliteEventRepository(manager)
    trial = TrialSpec(
        trial_id="trial_overlay_error",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="broken",
                    extractor_chain=ExtractorChainSpec(extractors=["broken"]),
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    transform_hash = resolve_task_stages(trial.task).output_transforms[0].transform_hash
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
            metadata=cast(TrialEventMetadata, {"provider": "fake", "model_id": "test"}),
            payload={"spec_hash": "inf_hash", "raw_text": "42"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.CANDIDATE_FAILED,
            candidate_id="candidate_1",
            stage=TimelineStage.EXTRACTION,
            status=RecordStatus.ERROR,
            metadata=cast(TrialEventMetadata, {"transform_hash": transform_hash}),
            error=ErrorRecord(
                code=ErrorCode.PARSE_ERROR,
                message="extractor boom",
                retryable=False,
                where=ErrorWhere.EXTRACTOR,
            ),
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
                    "projection_version": "v2",
                },
            ),
        ),
    ]:
        event_repo.append_event(event)

    generation_record = repo.materialize_trial_record(trial.spec_hash)
    overlay_record = repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
    )

    assert generation_record.status == RecordStatus.OK
    assert generation_record.candidates[0].status == RecordStatus.OK
    assert overlay_record.status == RecordStatus.ERROR
    assert overlay_record.candidates[0].status == RecordStatus.ERROR
    assert repo.has_trial(trial.spec_hash, transform_hash=transform_hash) is False


def test_projection_repo_iter_trial_summaries_filters_to_selected_overlay(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/proj_summary_filter.db")
    manager.initialize()

    repo = SqliteProjectionRepository(manager)
    event_repo = SqliteEventRepository(manager)
    trial = TrialSpec(
        trial_id="trial_summary_filter",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="broken",
                    extractor_chain=ExtractorChainSpec(extractors=["broken"]),
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    transform_hash = resolve_task_stages(trial.task).output_transforms[0].transform_hash
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
            metadata=cast(TrialEventMetadata, {"provider": "fake", "model_id": "test"}),
            payload={"spec_hash": "inf_hash", "raw_text": "42"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.CANDIDATE_FAILED,
            candidate_id="candidate_1",
            stage=TimelineStage.EXTRACTION,
            status=RecordStatus.ERROR,
            metadata=cast(TrialEventMetadata, {"transform_hash": transform_hash}),
            error=ErrorRecord(
                code=ErrorCode.PARSE_ERROR,
                message="extractor boom",
                retryable=False,
                where=ErrorWhere.EXTRACTOR,
            ),
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
                    "projection_version": "v2",
                },
            ),
        ),
    ]:
        event_repo.append_event(event)

    repo.materialize_trial_record(trial.spec_hash)
    repo.materialize_trial_record(trial.spec_hash, transform_hash=transform_hash)

    generation_rows = list(repo.iter_trial_summaries(trial_hashes=[trial.spec_hash]))
    overlay_rows = list(
        repo.iter_trial_summaries(
            trial_hashes=[trial.spec_hash],
            transform_hash=transform_hash,
        )
    )

    assert generation_rows == [
        generation_rows[0].model_copy(update={"status": RecordStatus.OK})
    ]
    assert overlay_rows == [
        overlay_rows[0].model_copy(update={"status": RecordStatus.ERROR})
    ]
