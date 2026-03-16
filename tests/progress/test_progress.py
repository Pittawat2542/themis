from __future__ import annotations

from datetime import datetime, timezone
import threading

import pytest


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


def test_run_progress_tracker_preserves_previous_error_fields_when_not_overridden():
    from themis.orchestration.run_manifest import (
        RunManifest,
        StageWorkItem,
        WorkItemStatus,
    )
    from themis.progress import ProgressConfig, RunProgressSnapshot, RunProgressTracker
    from themis.progress.models import StageProgressSnapshot
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
        def get_progress_snapshot(self, run_id: str) -> RunProgressSnapshot | None:
            assert run_id == "run-1"
            return snapshot

        def update_work_item(self, *args, **kwargs) -> None:
            del args, kwargs

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
    tracker = RunProgressTracker(
        RunManifest(
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
                    status=WorkItemStatus.FAILED,
                    trial_hash="trial-1",
                    candidate_index=0,
                    candidate_id="candidate-1",
                    last_error_code="existing-code",
                    last_error_message="existing-message",
                )
            ],
        ),
        FakeRunManifestRepository(),
        ProgressConfig(enabled=False),
    )

    tracker.mark_finished("work-1", status=WorkItemStatus.COMPLETED)

    updated = tracker._work_items["work-1"]
    assert updated.last_error_code == "existing-code"
    assert updated.last_error_message == "existing-message"


def test_run_progress_tracker_serializes_concurrent_work_item_finishes(monkeypatch):
    from themis.orchestration.run_manifest import (
        RunManifest,
        StageWorkItem,
        WorkItemStatus,
    )
    from themis.progress import ProgressConfig, RunProgressTracker
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
        def get_progress_snapshot(self, run_id: str):
            assert run_id == "run-1"
            return None

        def update_work_item(self, *args, **kwargs) -> None:
            del args, kwargs

    tracker = RunProgressTracker(
        RunManifest(
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
                    status=WorkItemStatus.RUNNING,
                    trial_hash="trial-1",
                    candidate_index=0,
                    candidate_id="candidate-1",
                ),
                StageWorkItem(
                    work_item_id="work-2",
                    stage=RunStage.GENERATION,
                    status=WorkItemStatus.RUNNING,
                    trial_hash="trial-1",
                    candidate_index=1,
                    candidate_id="candidate-2",
                ),
            ],
        ),
        FakeRunManifestRepository(),
        ProgressConfig(enabled=False),
    )

    first_snapshot_captured = threading.Event()
    second_snapshot_captured = threading.Event()
    capture_order_lock = threading.Lock()
    capture_count = 0

    def racing_update(self, stage, previous_status, next_status, *, started_at):
        nonlocal capture_count
        captured_snapshot = self.snapshot
        with capture_order_lock:
            capture_count += 1
            capture_position = capture_count
        if capture_position == 1:
            first_snapshot_captured.set()
            second_snapshot_captured.wait(timeout=0.2)
        else:
            first_snapshot_captured.wait(timeout=0.2)
            second_snapshot_captured.set()
        counts = captured_snapshot.stage_counts[stage]
        updates = {
            "pending_items": counts.pending_items,
            "running_items": counts.running_items,
            "completed_items": counts.completed_items,
            "failed_items": counts.failed_items,
            "skipped_items": counts.skipped_items,
        }
        self._decrement_status_bucket(updates, previous_status)
        self._increment_status_bucket(updates, next_status)
        stage_counts = dict(captured_snapshot.stage_counts)
        stage_counts[stage] = counts.model_copy(update=updates)
        started_candidates = [
            item.started_at
            for item in self._work_items.values()
            if getattr(item, "started_at", None) is not None
        ]
        self.snapshot = captured_snapshot.model_copy(
            update={
                "processed_items": captured_snapshot.processed_items
                - self._processed_delta(previous_status)
                + self._processed_delta(next_status),
                "remaining_items": captured_snapshot.remaining_items
                - self._remaining_delta(previous_status)
                + self._remaining_delta(next_status),
                "in_flight_items": captured_snapshot.in_flight_items
                - self._in_flight_delta(previous_status)
                + self._in_flight_delta(next_status),
                "stage_counts": stage_counts,
                "started_at": min(
                    [
                        candidate
                        for candidate in started_candidates
                        if candidate is not None
                    ],
                    default=started_at or captured_snapshot.started_at,
                ),
                "active_stage": self._active_stage(stage_counts),
                "ended_at": None,
            }
        )

    monkeypatch.setattr(
        RunProgressTracker,
        "_update_snapshot_for_transition",
        racing_update,
    )

    threads = [
        threading.Thread(
            target=tracker.mark_finished,
            args=(work_item_id,),
            kwargs={"status": WorkItemStatus.COMPLETED},
        )
        for work_item_id in ("work-1", "work-2")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tracker.snapshot.processed_items == 2
    assert tracker.snapshot.remaining_items == 0
    assert tracker.snapshot.in_flight_items == 0


def test_run_progress_tracker_lookup_errors_include_context():
    from themis.orchestration.run_manifest import RunManifest
    from themis.progress import ProgressConfig, RunProgressSnapshot, RunProgressTracker
    from themis.progress.models import StageProgressSnapshot
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
        def get_progress_snapshot(self, run_id: str) -> RunProgressSnapshot | None:
            assert run_id == "run-1"
            return snapshot

        def update_work_item(self, *args, **kwargs) -> None:
            del args, kwargs

    snapshot = RunProgressSnapshot(
        run_id="run-1",
        backend_kind="local",
        active_stage=RunStage.GENERATION,
        processed_items=0,
        remaining_items=0,
        in_flight_items=0,
        stage_counts={
            RunStage.GENERATION: StageProgressSnapshot(
                stage=RunStage.GENERATION,
                total_items=0,
                pending_items=0,
                running_items=0,
                completed_items=0,
                failed_items=0,
                skipped_items=0,
            )
        },
    )
    tracker = RunProgressTracker(
        RunManifest(
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
        ),
        FakeRunManifestRepository(),
        ProgressConfig(enabled=False),
    )

    with pytest.raises(KeyError, match="trial_hash='trial-1'.*candidate_index=0"):
        tracker.generation_work_item_id("trial-1", 0)
