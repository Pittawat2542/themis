"""Persistence helpers for run manifests and stage work items."""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import TypeAdapter

from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.progress.models import RunProgressSnapshot, StageProgressSnapshot
from themis.types.enums import RunStage

_STAGE_WORK_ITEMS_ADAPTER: TypeAdapter[list[StageWorkItem]] = TypeAdapter(
    list[StageWorkItem]
)
_TERMINAL_WORK_ITEM_STATUSES = {
    WorkItemStatus.COMPLETED,
    WorkItemStatus.FAILED,
    WorkItemStatus.SKIPPED,
}


def _parse_datetime(raw_value: str | None) -> datetime | None:
    if raw_value is None:
        return None
    return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))


def _merged_work_item_status(
    planned_status: WorkItemStatus,
    stored_status: WorkItemStatus,
) -> WorkItemStatus:
    if planned_status in _TERMINAL_WORK_ITEM_STATUSES:
        return planned_status
    if stored_status in _TERMINAL_WORK_ITEM_STATUSES:
        return stored_status
    if (
        planned_status == WorkItemStatus.PENDING
        and stored_status == WorkItemStatus.RUNNING
    ):
        return WorkItemStatus.RUNNING
    return planned_status


class RunManifestRepository:
    """Store and retrieve persisted run manifests backed by the active SQL manager."""

    def __init__(self, manager) -> None:
        self.manager = manager

    def save_manifest(self, manifest: RunManifest) -> None:
        """Persist one run manifest and its normalized stage work items."""
        with self.manager.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO run_manifests (
                        run_id,
                        backend_kind,
                        project_spec_json,
                        experiment_spec_json,
                        manifest_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        backend_kind=excluded.backend_kind,
                        project_spec_json=excluded.project_spec_json,
                        experiment_spec_json=excluded.experiment_spec_json,
                        manifest_json=excluded.manifest_json,
                        created_at=excluded.created_at
                    """,
                    (
                        manifest.run_id,
                        manifest.backend_kind,
                        json.dumps(manifest.project_spec.model_dump(mode="json"))
                        if manifest.project_spec is not None
                        else None,
                        json.dumps(manifest.experiment_spec.model_dump(mode="json")),
                        json.dumps(manifest.model_dump(mode="json")),
                        manifest.created_at.isoformat(),
                    ),
                )
                conn.execute(
                    "DELETE FROM stage_work_items WHERE run_id = ?",
                    (manifest.run_id,),
                )
                for item in manifest.work_items:
                    conn.execute(
                        """
                        INSERT INTO stage_work_items (
                            work_item_id,
                            run_id,
                            stage,
                            status,
                            trial_hash,
                            candidate_index,
                            candidate_id,
                            transform_hash,
                            evaluation_hash,
                            attempt_count,
                            lease_owner,
                            lease_expires_at,
                            external_job_id,
                            artifact_refs_json,
                            started_at,
                            ended_at,
                            last_error_code,
                            last_error_message
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item.work_item_id,
                            manifest.run_id,
                            item.stage,
                            item.status.value,
                            item.trial_hash,
                            item.candidate_index,
                            item.candidate_id,
                            item.transform_hash,
                            item.evaluation_hash,
                            item.attempt_count,
                            item.lease_owner,
                            item.lease_expires_at.isoformat()
                            if item.lease_expires_at is not None
                            else None,
                            item.external_job_id,
                            json.dumps(item.artifact_refs),
                            item.started_at.isoformat()
                            if item.started_at is not None
                            else None,
                            item.ended_at.isoformat()
                            if item.ended_at is not None
                            else None,
                            item.last_error_code,
                            item.last_error_message,
                        ),
                    )

    def reconcile_manifest(
        self,
        manifest: RunManifest,
        *,
        stored_manifest: RunManifest | None = None,
    ) -> RunManifest:
        if stored_manifest is None:
            stored_manifest = self.get_manifest(manifest.run_id)
        if stored_manifest is None:
            return manifest

        stored_items = {item.work_item_id: item for item in stored_manifest.work_items}
        merged_items: list[StageWorkItem] = []
        for planned_item in manifest.work_items:
            stored_item = stored_items.get(planned_item.work_item_id)
            if stored_item is None:
                merged_items.append(planned_item)
                continue
            merged_items.append(
                planned_item.model_copy(
                    update={
                        "status": _merged_work_item_status(
                            planned_item.status,
                            stored_item.status,
                        ),
                        "attempt_count": stored_item.attempt_count,
                        "lease_owner": stored_item.lease_owner,
                        "lease_expires_at": stored_item.lease_expires_at,
                        "external_job_id": stored_item.external_job_id,
                        "artifact_refs": list(stored_item.artifact_refs),
                        "started_at": stored_item.started_at,
                        "ended_at": stored_item.ended_at,
                        "last_error_code": stored_item.last_error_code,
                        "last_error_message": stored_item.last_error_message,
                    }
                )
            )
        return manifest.model_copy(
            update={
                "created_at": stored_manifest.created_at,
                "work_items": merged_items,
            }
        )

    def get_manifest(self, run_id: str) -> RunManifest | None:
        """Load one manifest plus its stage work items by deterministic run ID."""
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT manifest_json
                FROM run_manifests
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            work_item_rows = conn.execute(
                """
                SELECT work_item_id, stage, status, trial_hash, candidate_index,
                       candidate_id, transform_hash, evaluation_hash, attempt_count,
                       lease_owner, lease_expires_at, external_job_id, artifact_refs_json,
                       started_at, ended_at, last_error_code, last_error_message
                FROM stage_work_items
                WHERE run_id = ?
                ORDER BY stage ASC, trial_hash ASC, candidate_index ASC, work_item_id ASC
                """,
                (run_id,),
            ).fetchall()

        manifest_payload = json.loads(row["manifest_json"])
        if work_item_rows:
            manifest_payload["work_items"] = _STAGE_WORK_ITEMS_ADAPTER.validate_python(
                [
                    {
                        "work_item_id": work_item_row["work_item_id"],
                        "stage": work_item_row["stage"],
                        "status": work_item_row["status"],
                        "trial_hash": work_item_row["trial_hash"],
                        "candidate_index": work_item_row["candidate_index"],
                        "candidate_id": work_item_row["candidate_id"],
                        "transform_hash": work_item_row["transform_hash"],
                        "evaluation_hash": work_item_row["evaluation_hash"],
                        "attempt_count": work_item_row["attempt_count"],
                        "lease_owner": work_item_row["lease_owner"],
                        "lease_expires_at": work_item_row["lease_expires_at"],
                        "external_job_id": work_item_row["external_job_id"],
                        "artifact_refs": json.loads(
                            work_item_row["artifact_refs_json"] or "[]"
                        ),
                        "started_at": work_item_row["started_at"],
                        "ended_at": work_item_row["ended_at"],
                        "last_error_code": work_item_row["last_error_code"],
                        "last_error_message": work_item_row["last_error_message"],
                    }
                    for work_item_row in work_item_rows
                ]
            )
        return RunManifest.model_validate(manifest_payload)

    def update_work_item(
        self,
        run_id: str,
        work_item_id: str,
        *,
        status: WorkItemStatus,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        attempt_count: int | None = None,
        last_error_code: str | None = None,
        last_error_message: str | None = None,
    ) -> None:
        with self.manager.get_connection() as conn:
            with conn:
                assignments = ["status = ?"]
                params: list[object] = [status.value]
                if started_at is not None:
                    assignments.append("started_at = ?")
                    params.append(started_at.isoformat())
                if ended_at is not None:
                    assignments.append("ended_at = ?")
                    params.append(ended_at.isoformat())
                if attempt_count is not None:
                    assignments.append("attempt_count = ?")
                    params.append(attempt_count)
                if last_error_code is not None:
                    assignments.append("last_error_code = ?")
                    params.append(last_error_code)
                if last_error_message is not None:
                    assignments.append("last_error_message = ?")
                    params.append(last_error_message)
                params.extend([run_id, work_item_id])
                conn.execute(
                    f"""
                    UPDATE stage_work_items
                    SET {", ".join(assignments)}
                    WHERE run_id = ? AND work_item_id = ?
                    """,
                    params,
                )

    def get_progress_snapshot(self, run_id: str) -> RunProgressSnapshot | None:
        with self.manager.get_connection() as conn:
            manifest_row = conn.execute(
                """
                SELECT run_id, backend_kind, created_at
                FROM run_manifests
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if manifest_row is None:
                return None
            stage_rows = conn.execute(
                """
                SELECT
                    stage,
                    COUNT(*) AS total_items,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_items,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running_items,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_items,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_items,
                    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped_items,
                    MIN(started_at) AS started_at,
                    MAX(ended_at) AS ended_at
                FROM stage_work_items
                WHERE run_id = ?
                GROUP BY stage
                """,
                (run_id,),
            ).fetchall()

        stage_counts: dict[RunStage, StageProgressSnapshot] = {}
        processed_items = 0
        remaining_items = 0
        in_flight_items = 0
        started_candidates: list[datetime] = []
        ended_candidates: list[datetime] = []
        rows_by_stage = {RunStage(row["stage"]): row for row in stage_rows}
        for stage in RunStage:
            row = rows_by_stage.get(stage)
            if row is None:
                continue
            pending_items = int(row["pending_items"] or 0)
            running_items = int(row["running_items"] or 0)
            completed_items = int(row["completed_items"] or 0)
            failed_items = int(row["failed_items"] or 0)
            skipped_items = int(row["skipped_items"] or 0)
            stage_started_at = _parse_datetime(row["started_at"])
            stage_ended_at = _parse_datetime(row["ended_at"])
            if stage_started_at is not None:
                started_candidates.append(stage_started_at)
            if stage_ended_at is not None:
                ended_candidates.append(stage_ended_at)
            stage_counts[stage] = StageProgressSnapshot(
                stage=stage,
                total_items=int(row["total_items"] or 0),
                pending_items=pending_items,
                running_items=running_items,
                completed_items=completed_items,
                failed_items=failed_items,
                skipped_items=skipped_items,
            )
            processed_items += completed_items + failed_items + skipped_items
            remaining_items += pending_items + running_items
            in_flight_items += running_items
        active_stage = next(
            (
                stage
                for stage in RunStage
                if stage in stage_counts and stage_counts[stage].running_items > 0
            ),
            None,
        )
        if active_stage is None:
            active_stage = next(
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
            )
        return RunProgressSnapshot(
            run_id=manifest_row["run_id"],
            backend_kind=manifest_row["backend_kind"],
            active_stage=active_stage,
            processed_items=processed_items,
            remaining_items=remaining_items,
            in_flight_items=in_flight_items,
            stage_counts=stage_counts,
            started_at=min(
                started_candidates,
                default=_parse_datetime(manifest_row["created_at"]),
            ),
            ended_at=max(ended_candidates)
            if remaining_items == 0 and ended_candidates
            else None,
        )
