"""Progress tracking and snapshot updates for persisted run manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading

from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.progress.bus import ProgressBus, ProgressEventType
from themis.progress.models import (
    ProgressConfig,
    ProgressRendererType,
    RunProgressSnapshot,
    StageProgressSnapshot,
)
from themis.progress.renderers import ProgressLogRenderer, RichProgressRenderer
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.types.enums import RunStage


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RunProgressTracker:
    """Persists work-item progress and fans out run snapshots to observers."""

    manifest: RunManifest
    manifest_repo: RunManifestRepository
    config: ProgressConfig | None = None
    allowed_stages: frozenset[RunStage] | None = None
    bus: ProgressBus = field(init=False)
    snapshot: RunProgressSnapshot = field(init=False)
    _lock: threading.RLock = field(init=False, default_factory=threading.RLock)
    _work_items: dict[str, StageWorkItem] = field(init=False, default_factory=dict)
    _generation_ids: dict[tuple[str, int], str] = field(
        init=False, default_factory=dict
    )
    _transform_ids: dict[tuple[str, int, str | None], str] = field(
        init=False, default_factory=dict
    )
    _evaluation_ids: dict[tuple[str, int, str | None], str] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.bus = ProgressBus()
        self.snapshot = (
            self.manifest_repo.get_progress_snapshot(self.manifest.run_id)
            or self._snapshot_from_manifest()
        )
        if self.allowed_stages is None:
            self.allowed_stages = frozenset(RunStage)
        if self.config is not None and self.config.enabled:
            callback = self.config.callback
            if callback is not None:
                self.bus.subscribe(lambda event: callback(event.snapshot))
            renderer = self.config.renderer
            if renderer is None and callback is None:
                renderer = ProgressRendererType.RICH
            if renderer == ProgressRendererType.LOG:
                self.bus.subscribe(ProgressLogRenderer(self.config.verbosity))
            elif renderer == ProgressRendererType.RICH:
                self.bus.subscribe(RichProgressRenderer())
        self._work_items = {
            item.work_item_id: item for item in self.manifest.work_items
        }
        self._generation_ids = {
            (item.trial_hash, item.candidate_index): item.work_item_id
            for item in self.manifest.work_items
            if item.stage == RunStage.GENERATION
        }
        self._transform_ids = {
            (
                item.trial_hash,
                item.candidate_index,
                item.transform_hash,
            ): item.work_item_id
            for item in self.manifest.work_items
            if item.stage == RunStage.TRANSFORM
        }
        self._evaluation_ids = {
            (
                item.trial_hash,
                item.candidate_index,
                item.evaluation_hash,
            ): item.work_item_id
            for item in self.manifest.work_items
            if item.stage == RunStage.EVALUATION
        }

    def generation_work_item_id(self, trial_hash: str, candidate_index: int) -> str:
        """Returns the generation work item ID for one trial candidate."""

        key = (trial_hash, candidate_index)
        if key not in self._generation_ids:
            raise KeyError(
                "Missing generation work item for "
                f"trial_hash='{trial_hash}', candidate_index={candidate_index}."
            )
        return self._generation_ids[key]

    def transform_work_item_id(
        self,
        trial_hash: str,
        candidate_index: int,
        transform_hash: str,
    ) -> str:
        """Returns the transform work item ID for a candidate overlay."""

        key = (trial_hash, candidate_index, transform_hash)
        if key not in self._transform_ids:
            raise KeyError(
                "Missing transform work item for "
                f"trial_hash='{trial_hash}', candidate_index={candidate_index}, "
                f"transform_hash='{transform_hash}'."
            )
        return self._transform_ids[key]

    def evaluation_work_item_id(
        self,
        trial_hash: str,
        candidate_index: int,
        evaluation_hash: str,
    ) -> str:
        """Returns the evaluation work item ID for a candidate overlay."""

        key = (trial_hash, candidate_index, evaluation_hash)
        if key not in self._evaluation_ids:
            raise KeyError(
                "Missing evaluation work item for "
                f"trial_hash='{trial_hash}', candidate_index={candidate_index}, "
                f"evaluation_hash='{evaluation_hash}'."
            )
        return self._evaluation_ids[key]

    def start_run(self) -> None:
        """Emits the initial run-started progress snapshot."""

        with self._lock:
            self._emit(ProgressEventType.RUN_STARTED)

    def stage_started(self) -> None:
        """Emits a snapshot after orchestration advances to a new stage."""

        with self._lock:
            self._emit(ProgressEventType.STAGE_STARTED)

    def mark_running(self, work_item_id: str) -> None:
        """Marks a work item as running and emits an updated snapshot."""

        with self._lock:
            started_at = _now_utc()
            self.manifest_repo.update_work_item(
                self.manifest.run_id,
                work_item_id,
                status=WorkItemStatus.RUNNING,
                started_at=started_at,
            )
            self._update_work_item(
                work_item_id,
                status=WorkItemStatus.RUNNING,
                started_at=started_at,
            )
            self._emit(ProgressEventType.WORK_ITEM_STARTED)

    def mark_finished(
        self,
        work_item_id: str,
        *,
        status: WorkItemStatus,
        last_error_code: str | None = None,
        last_error_message: str | None = None,
    ) -> None:
        """Marks a work item terminal and emits an updated snapshot."""

        with self._lock:
            ended_at = _now_utc()
            self.manifest_repo.update_work_item(
                self.manifest.run_id,
                work_item_id,
                status=status,
                ended_at=ended_at,
                last_error_code=last_error_code,
                last_error_message=last_error_message,
            )
            self._update_work_item(
                work_item_id,
                status=status,
                ended_at=ended_at,
                last_error_code=last_error_code,
                last_error_message=last_error_message,
            )
            self._emit(ProgressEventType.WORK_ITEM_FINISHED)

    def finish_run(self) -> None:
        """Finalizes the run snapshot and emits the run-finished event."""

        with self._lock:
            ended_at = None
            if self.snapshot.remaining_items == 0:
                ended_candidates: list[datetime] = []
                for item in self._work_items.values():
                    if item.ended_at is not None:
                        ended_candidates.append(item.ended_at)
                ended_at = max(ended_candidates) if ended_candidates else _now_utc()
            self.snapshot = self.snapshot.model_copy(update={"ended_at": ended_at})
            self._emit(ProgressEventType.RUN_FINISHED)

    def _emit(self, event_type: ProgressEventType) -> None:
        self.bus.emit(event_type, snapshot=self.snapshot)

    def _snapshot_from_manifest(self) -> RunProgressSnapshot:
        stage_counts: dict[RunStage, StageProgressSnapshot] = {}
        started_candidates = [
            item.started_at
            for item in self.manifest.work_items
            if item.started_at is not None
        ]
        ended_candidates: list[datetime] = [
            item.ended_at
            for item in self.manifest.work_items
            if item.ended_at is not None
        ]
        processed_items = 0
        remaining_items = 0
        in_flight_items = 0
        for stage in RunStage:
            items = [item for item in self.manifest.work_items if item.stage == stage]
            if not items:
                continue
            pending_items = sum(
                1 for item in items if item.status == WorkItemStatus.PENDING
            )
            running_items = sum(
                1 for item in items if item.status == WorkItemStatus.RUNNING
            )
            completed_items = sum(
                1 for item in items if item.status == WorkItemStatus.COMPLETED
            )
            failed_items = sum(
                1 for item in items if item.status == WorkItemStatus.FAILED
            )
            skipped_items = sum(
                1 for item in items if item.status == WorkItemStatus.SKIPPED
            )
            stage_counts[stage] = StageProgressSnapshot(
                stage=stage,
                total_items=len(items),
                pending_items=pending_items,
                running_items=running_items,
                completed_items=completed_items,
                failed_items=failed_items,
                skipped_items=skipped_items,
            )
            processed_items += completed_items + failed_items + skipped_items
            remaining_items += pending_items + running_items
            in_flight_items += running_items
        return RunProgressSnapshot(
            run_id=self.manifest.run_id,
            backend_kind=self.manifest.backend_kind,
            active_stage=self._active_stage(stage_counts),
            processed_items=processed_items,
            remaining_items=remaining_items,
            in_flight_items=in_flight_items,
            stage_counts=stage_counts,
            started_at=min(started_candidates, default=self.manifest.created_at),
            ended_at=max(ended_candidates)
            if remaining_items == 0 and ended_candidates
            else None,
        )

    def _update_work_item(
        self,
        work_item_id: str,
        *,
        status: WorkItemStatus,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        last_error_code: str | None = None,
        last_error_message: str | None = None,
    ) -> None:
        work_item = self._work_items[work_item_id]
        updated_item = work_item.model_copy(
            update={
                "status": status,
                "started_at": started_at
                if started_at is not None
                else work_item.started_at,
                "ended_at": ended_at if ended_at is not None else work_item.ended_at,
                **(
                    {"last_error_code": last_error_code}
                    if last_error_code is not None
                    else {}
                ),
                **(
                    {"last_error_message": last_error_message}
                    if last_error_message is not None
                    else {}
                ),
            }
        )
        self._work_items[work_item_id] = updated_item
        self._update_snapshot_for_transition(
            updated_item.stage,
            work_item.status,
            status,
            started_at=updated_item.started_at,
        )

    def _update_snapshot_for_transition(
        self,
        stage: RunStage,
        previous_status: WorkItemStatus,
        next_status: WorkItemStatus,
        *,
        started_at: datetime | None,
    ) -> None:
        counts = self.snapshot.stage_counts[stage]
        updates = {
            "pending_items": counts.pending_items,
            "running_items": counts.running_items,
            "completed_items": counts.completed_items,
            "failed_items": counts.failed_items,
            "skipped_items": counts.skipped_items,
        }
        self._decrement_status_bucket(updates, previous_status)
        self._increment_status_bucket(updates, next_status)
        stage_counts = dict(self.snapshot.stage_counts)
        stage_counts[stage] = counts.model_copy(update=updates)
        started_candidates = [
            item.started_at
            for item in self._work_items.values()
            if getattr(item, "started_at", None) is not None
        ]
        self.snapshot = self.snapshot.model_copy(
            update={
                "processed_items": self.snapshot.processed_items
                - self._processed_delta(previous_status)
                + self._processed_delta(next_status),
                "remaining_items": self.snapshot.remaining_items
                - self._remaining_delta(previous_status)
                + self._remaining_delta(next_status),
                "in_flight_items": self.snapshot.in_flight_items
                - self._in_flight_delta(previous_status)
                + self._in_flight_delta(next_status),
                "stage_counts": stage_counts,
                "started_at": min(
                    [
                        candidate
                        for candidate in started_candidates
                        if candidate is not None
                    ],
                    default=started_at or self.snapshot.started_at,
                ),
                "active_stage": self._active_stage(stage_counts),
                "ended_at": None,
            }
        )

    def _increment_status_bucket(
        self,
        updates: dict[str, int],
        status: WorkItemStatus,
    ) -> None:
        bucket_name = self._bucket_name(status)
        if bucket_name is not None:
            updates[bucket_name] += 1

    def _decrement_status_bucket(
        self,
        updates: dict[str, int],
        status: WorkItemStatus,
    ) -> None:
        bucket_name = self._bucket_name(status)
        if bucket_name is not None:
            updates[bucket_name] -= 1

    def _bucket_name(self, status: WorkItemStatus) -> str | None:
        if status == WorkItemStatus.PENDING:
            return "pending_items"
        if status == WorkItemStatus.RUNNING:
            return "running_items"
        if status == WorkItemStatus.COMPLETED:
            return "completed_items"
        if status == WorkItemStatus.FAILED:
            return "failed_items"
        if status == WorkItemStatus.SKIPPED:
            return "skipped_items"
        return None

    def _processed_delta(self, status: WorkItemStatus) -> int:
        return int(
            status
            in {
                WorkItemStatus.COMPLETED,
                WorkItemStatus.FAILED,
                WorkItemStatus.SKIPPED,
            }
        )

    def _remaining_delta(self, status: WorkItemStatus) -> int:
        return int(status in {WorkItemStatus.PENDING, WorkItemStatus.RUNNING})

    def _in_flight_delta(self, status: WorkItemStatus) -> int:
        return int(status == WorkItemStatus.RUNNING)

    def _active_stage(
        self,
        stage_counts: dict[RunStage, StageProgressSnapshot],
    ) -> RunStage | None:
        return next(
            (
                stage
                for stage in RunStage
                if stage in stage_counts and stage_counts[stage].running_items > 0
            ),
            next(
                (
                    stage
                    for stage in RunStage
                    if stage in stage_counts
                    and (
                        stage_counts[stage].pending_items
                        or stage_counts[stage].running_items
                    )
                ),
                None,
            ),
        )
