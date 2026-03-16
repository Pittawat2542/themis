from __future__ import annotations

from datetime import datetime, timezone


def test_progress_bus_emits_structured_events_to_subscribers():
    from themis.progress import (
        ProgressBus,
        ProgressEventType,
        RunProgressSnapshot,
        StageProgressSnapshot,
    )
    from themis.types.enums import RunStage

    bus = ProgressBus()
    seen = []
    snapshot = RunProgressSnapshot(
        run_id="run-1",
        backend_kind="local",
        active_stage=RunStage.GENERATION,
        processed_items=1,
        remaining_items=2,
        in_flight_items=1,
        stage_counts={
            RunStage.GENERATION: StageProgressSnapshot(
                stage=RunStage.GENERATION,
                total_items=3,
                pending_items=1,
                running_items=1,
                completed_items=1,
                failed_items=0,
                skipped_items=0,
            )
        },
    )
    bus.subscribe(seen.append)

    bus.emit(ProgressEventType.RUN_STARTED, snapshot=snapshot)

    assert len(seen) == 1
    assert seen[0].event_type == ProgressEventType.RUN_STARTED
    assert seen[0].snapshot == snapshot


def test_progress_config_supports_callback_and_renderer_defaults():
    from themis.progress import ProgressConfig, ProgressRendererType, ProgressVerbosity

    snapshots = []
    config = ProgressConfig(callback=snapshots.append)

    assert config.enabled is True
    assert config.renderer is None
    assert config.verbosity == ProgressVerbosity.NORMAL
    assert config.callback is not None

    explicit_renderer = ProgressConfig(
        callback=snapshots.append,
        renderer=ProgressRendererType.LOG,
    )

    assert explicit_renderer.renderer == ProgressRendererType.LOG


def test_run_progress_snapshot_tracks_terminal_timestamp():
    from themis.progress import RunProgressSnapshot

    ended_at = datetime.now(timezone.utc)
    snapshot = RunProgressSnapshot(
        run_id="run-1",
        backend_kind="batch",
        active_stage=None,
        processed_items=4,
        remaining_items=0,
        in_flight_items=0,
        stage_counts={},
        ended_at=ended_at,
    )

    assert snapshot.ended_at == ended_at


def test_run_progress_tracker_uses_cached_snapshots_for_work_item_events():
    from themis.orchestration.run_manifest import (
        RunManifest,
        StageWorkItem,
        WorkItemStatus,
    )
    from themis.progress import (
        ProgressConfig,
        RunProgressSnapshot,
        RunProgressTracker,
        StageProgressSnapshot,
    )
    from themis.specs.experiment import (
        ExperimentSpec,
        InferenceGridSpec,
        InferenceParamsSpec,
        PromptTemplateSpec,
    )
    from themis.specs.foundational import (
        DatasetSpec,
        GenerationSpec,
        ModelSpec,
        TaskSpec,
    )
    from themis.types.enums import DatasetSource, RunStage

    class FakeRunManifestRepository:
        def __init__(self, snapshot: RunProgressSnapshot) -> None:
            self.snapshot = snapshot
            self.snapshot_calls = 0
            self.updates: list[tuple[str, WorkItemStatus]] = []

        def get_progress_snapshot(self, run_id: str) -> RunProgressSnapshot | None:
            assert run_id == "run-1"
            self.snapshot_calls += 1
            if self.snapshot_calls > 1:
                raise AssertionError("tracker should not reload snapshots per event")
            return self.snapshot

        def update_work_item(
            self,
            run_id: str,
            work_item_id: str,
            *,
            status: WorkItemStatus,
            started_at=None,
            ended_at=None,
            attempt_count=None,
            last_error_code=None,
            last_error_message=None,
        ) -> None:
            del started_at, ended_at, attempt_count, last_error_code, last_error_message
            assert run_id == "run-1"
            self.updates.append((work_item_id, status))

    snapshot = RunProgressSnapshot(
        run_id="run-1",
        backend_kind="local",
        active_stage=RunStage.GENERATION,
        processed_items=0,
        remaining_items=1,
        in_flight_items=0,
        stage_counts={
            RunStage.GENERATION: StageProgressSnapshot(
                stage=RunStage.GENERATION,
                total_items=1,
                pending_items=1,
                running_items=0,
                completed_items=0,
                failed_items=0,
                skipped_items=0,
            )
        },
    )
    repo = FakeRunManifestRepository(snapshot)
    manifest = RunManifest(
        run_id="run-1",
        backend_kind="local",
        experiment_spec=ExperimentSpec(
            models=[ModelSpec(model_id="mock-model", provider="mock")],
            tasks=[
                TaskSpec(
                    task_id="task",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                )
            ],
            prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
            inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        ),
        work_items=[
            StageWorkItem(
                work_item_id="work-1",
                stage=RunStage.GENERATION,
                status=WorkItemStatus.PENDING,
                trial_hash="trial-1",
                candidate_index=0,
                candidate_id="candidate-1",
            )
        ],
    )

    tracker = RunProgressTracker(
        manifest,
        repo,
        ProgressConfig(enabled=False),
    )

    assert repo.snapshot_calls == 1

    tracker.mark_running("work-1")
    tracker.mark_finished("work-1", status=WorkItemStatus.COMPLETED)

    assert repo.snapshot_calls == 1
    assert repo.updates == [
        ("work-1", WorkItemStatus.RUNNING),
        ("work-1", WorkItemStatus.COMPLETED),
    ]
