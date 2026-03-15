"""Internal JSON and row-hydration helpers for SQLite projection reads."""

from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from themis.errors import StorageError
from themis.records.error import ErrorRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.timeline import TimelineStageRecord
from themis.storage.artifact_store import ArtifactStore
from themis.storage._protocols import StorageRow
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import ArtifactRef, TimelineStage
from themis.types.json_types import JSONDict, JSONList, JSONValueType
from themis.types.json_validation import format_validation_error

_JSON_DICT_ADAPTER: TypeAdapter[JSONDict] = TypeAdapter(JSONDict)
_JSON_LIST_ADAPTER: TypeAdapter[JSONList] = TypeAdapter(JSONList)
_JSON_VALUE_ADAPTER: TypeAdapter[JSONValueType] = TypeAdapter(JSONValueType)


class ProjectionCodecs:
    """Hydrates persisted JSON payloads and row records into typed models."""

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def load_judge_audit(self, artifact_hashes: list[str]) -> JudgeAuditTrail | None:
        if self.artifact_store is None or not artifact_hashes:
            return None

        trails: list[JudgeAuditTrail] = []
        for artifact_hash in artifact_hashes:
            try:
                payload = self.artifact_store.read_json(artifact_hash)
                trails.append(JudgeAuditTrail.model_validate(payload))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise self.storage_read_error(
                    "artifacts.judge_audit",
                    f"artifact_hash={artifact_hash}",
                    exc,
                ) from exc
        if not trails:
            return None
        if len(trails) == 1:
            return trails[0]

        judge_calls = []
        for trail in trails:
            judge_calls.extend(trail.judge_calls)
        return JudgeAuditTrail(
            spec_hash=trails[-1].spec_hash,
            candidate_hash=trails[-1].candidate_hash,
            judge_calls=judge_calls,
        )

    def decode_json_column(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: object | None,
    ) -> object | None:
        if raw_value is None:
            return default
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise self.storage_read_error(label, context, exc) from exc

    def load_json_dict(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: JSONDict,
    ) -> JSONDict:
        decoded = self.decode_json_column(
            raw_value,
            label=label,
            context=context,
            default=default,
        )
        try:
            return _JSON_DICT_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self.storage_read_error(label, context, exc) from exc

    def load_json_list(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: JSONList,
    ) -> JSONList:
        decoded = self.decode_json_column(
            raw_value,
            label=label,
            context=context,
            default=default,
        )
        try:
            return _JSON_LIST_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self.storage_read_error(label, context, exc) from exc

    def load_json_value(
        self,
        raw_value: str | None,
        *,
        label: str,
        context: str,
        default: JSONValueType | None,
    ) -> JSONValueType | None:
        decoded = self.decode_json_column(
            raw_value,
            label=label,
            context=context,
            default=default,
        )
        if decoded is None:
            return None
        try:
            return _JSON_VALUE_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self.storage_read_error(label, context, exc) from exc

    def load_timeline_stage_record(self, row: StorageRow) -> TimelineStageRecord:
        context = (
            f"trial_hash={row['trial_hash']}, record_id={row['record_id']}, "
            f"stage={row['stage_name']}, overlay_key={row['overlay_key']}"
        )
        metadata = self.load_json_dict(
            row["metadata_json"],
            label="record_timeline.metadata_json",
            context=context,
            default={},
        )
        artifact_items = self.load_json_list(
            row["artifacts_json"],
            label="record_timeline.artifacts_json",
            context=context,
            default=[],
        )
        error_payload = self.decode_json_column(
            row["error_json"],
            label="record_timeline.error_json",
            context=context,
            default=None,
        )
        try:
            artifact_refs = [
                ArtifactRef.model_validate(item) for item in artifact_items
            ]
        except ValidationError as exc:
            raise self.storage_read_error(
                "record_timeline.artifacts_json", context, exc
            ) from exc
        try:
            error = (
                ErrorRecord.model_validate(error_payload)
                if error_payload is not None
                else None
            )
        except ValidationError as exc:
            raise self.storage_read_error(
                "record_timeline.error_json", context, exc
            ) from exc
        return TimelineStageRecord(
            stage=TimelineStage(row["stage_name"]),
            status=RecordStatus(row["status"]),
            component_id=row["component_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_ms=row["duration_ms"],
            metadata=metadata if isinstance(metadata, dict) else {},
            artifact_refs=artifact_refs,
            error=error,
        )

    def storage_read_error(
        self,
        label: str,
        context: str,
        exc: ValidationError | json.JSONDecodeError,
    ) -> StorageError:
        detail = (
            format_validation_error(exc)
            if isinstance(exc, ValidationError)
            else exc.msg
        )
        return StorageError(
            code=ErrorCode.STORAGE_READ,
            message=f"Failed to hydrate {label} ({context}): {detail}",
        )
