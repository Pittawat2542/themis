"""SQLite-backed append-only repository for trial lifecycle events."""

from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from themis.errors import StorageError
from themis.records.error import ErrorRecord
from themis.specs.base import SpecBase
from themis.storage._protocols import (
    StorageConnection,
    StorageConnectionManager,
    StorageRow,
)
from themis.types.enums import ErrorCode
from themis.types.events import (
    ArtifactRef,
    ArtifactRole,
    ProjectionCompletedEventMetadata,
    TrialEvent,
    TrialEventType,
    parse_trial_event_metadata,
)
from themis.types.json_types import JSONDict, JSONList, JSONValueType
from themis.types.json_validation import format_validation_error

_JSON_DICT_ADAPTER: TypeAdapter[JSONDict] = TypeAdapter(JSONDict)
_JSON_LIST_ADAPTER: TypeAdapter[JSONList] = TypeAdapter(JSONList)
_JSON_VALUE_ADAPTER: TypeAdapter[JSONValueType] = TypeAdapter(JSONValueType)


class SqliteEventRepository:
    """Append-only repository for typed trial lifecycle events."""

    def __init__(self, manager: StorageConnectionManager):
        self.manager = manager

    def save_spec(self, spec: SpecBase, conn: StorageConnection | None = None) -> None:
        """Persist or update a canonical serialized spec."""
        if conn is None:
            with self.manager.get_connection() as local_conn:
                with local_conn:
                    self.save_spec(spec, conn=local_conn)
            return

        canonical_json = json.dumps(
            spec.model_dump(mode="json"),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = conn.execute(
            """
            SELECT canonical_hash, canonical_json
            FROM specs
            WHERE spec_hash = ?
            """,
            (spec.spec_hash,),
        ).fetchone()
        if existing is not None:
            existing_canonical_hash = self._persisted_spec_canonical_hash(
                spec,
                row=existing,
            )
            if existing_canonical_hash != spec.canonical_hash:
                raise StorageError(
                    code=ErrorCode.STORAGE_WRITE,
                    message=(
                        "short hash collision for persisted spec identity: "
                        f"spec_hash '{spec.spec_hash}' is already bound to a "
                        "different canonical payload"
                    ),
                    details={
                        "spec_hash": spec.spec_hash,
                        "existing_canonical_hash": existing_canonical_hash,
                        "incoming_canonical_hash": spec.canonical_hash,
                        "spec_type": spec.__class__.__name__,
                    },
                )

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
            ON CONFLICT(spec_hash) DO UPDATE SET
                canonical_hash=excluded.canonical_hash,
                spec_type=excluded.spec_type,
                schema_version=excluded.schema_version,
                canonical_json=excluded.canonical_json
            """,
            (
                spec.spec_hash,
                spec.canonical_hash,
                spec.__class__.__name__,
                str(spec.schema_version),
                canonical_json,
            ),
        )

    def append_event(
        self, event: TrialEvent, conn: StorageConnection | None = None
    ) -> None:
        """Append one typed lifecycle event to the event log."""
        if conn is None:
            with self.manager.get_connection() as local_conn:
                with local_conn:
                    self.append_event(event, conn=local_conn)
            return

        payload_json = self._payload_json_for_event(event)
        conn.execute(
            """
            INSERT INTO trial_events (
                trial_hash,
                event_seq,
                event_id,
                candidate_id,
                event_type,
                stage,
                status,
                event_ts,
                metadata_json,
                payload_json,
                artifact_refs_json,
                error_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.trial_hash,
                event.event_seq,
                event.event_id,
                event.candidate_id,
                event.event_type.value,
                event.stage.value if event.stage is not None else None,
                event.status.value if event.status is not None else None,
                event.event_ts.isoformat(),
                json.dumps(event.metadata.as_dict())
                if event.metadata.as_dict()
                else None,
                payload_json,
                json.dumps(
                    [
                        artifact.model_dump(mode="json")
                        for artifact in event.artifact_refs
                    ]
                )
                if event.artifact_refs
                else None,
                json.dumps(event.error.model_dump(mode="json"))
                if event.error is not None
                else None,
            ),
        )

    def _payload_json_for_event(self, event: TrialEvent) -> str | None:
        if event.payload is None:
            return None
        if self._payload_persisted_via_artifact(event):
            return None
        return json.dumps(event.payload)

    def _payload_persisted_via_artifact(self, event: TrialEvent) -> bool:
        if not event.artifact_refs or event.stage is None:
            return False
        expected_roles = {
            TrialEventType.ITEM_LOADED: {ArtifactRole.ITEM_PAYLOAD},
            TrialEventType.INFERENCE_COMPLETED: {ArtifactRole.INFERENCE_OUTPUT},
            TrialEventType.EXTRACTION_COMPLETED: {ArtifactRole.EXTRACTION_OUTPUT},
            TrialEventType.EVALUATION_COMPLETED: {ArtifactRole.EVALUATION_OUTPUT},
        }.get(event.event_type, set())
        return any(
            artifact.role in expected_roles
            and self._artifact_is_indexed(artifact.artifact_hash)
            for artifact in event.artifact_refs
        )

    def _artifact_is_indexed(self, artifact_hash: str) -> bool:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM artifacts
                WHERE artifact_hash = ?
                LIMIT 1
                """,
                (artifact_hash,),
            ).fetchone()
        return row is not None

    def get_events(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> list[TrialEvent]:
        """Load ordered events for a trial or one candidate within that trial."""
        with self.manager.get_connection() as conn:
            if candidate_id is None:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM trial_events
                    WHERE trial_hash = ?
                    ORDER BY event_seq ASC
                    """,
                    (trial_hash,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM trial_events
                    WHERE trial_hash = ? AND candidate_id = ?
                    ORDER BY event_seq ASC
                    """,
                    (trial_hash, candidate_id),
                ).fetchall()

        events: list[TrialEvent] = []
        for row in rows:
            metadata = self._load_json_dict(
                row["metadata_json"],
                label="trial_events.metadata_json",
                trial_hash=row["trial_hash"],
                candidate_id=row["candidate_id"],
            )
            try:
                parsed_metadata = parse_trial_event_metadata(
                    event_type=row["event_type"],
                    metadata=metadata,
                )
            except ValidationError as exc:
                raise self._storage_read_error(
                    "trial_events.metadata_json",
                    row["trial_hash"],
                    row["candidate_id"],
                    exc,
                ) from exc
            payload = self._load_json_value(
                row["payload_json"],
                label="trial_events.payload_json",
                trial_hash=row["trial_hash"],
                candidate_id=row["candidate_id"],
                default=None,
            )
            artifact_items = self._load_json_list(
                row["artifact_refs_json"],
                label="trial_events.artifact_refs_json",
                trial_hash=row["trial_hash"],
                candidate_id=row["candidate_id"],
            )
            error_payload = self._decode_json_column(
                row["error_json"],
                label="trial_events.error_json",
                trial_hash=row["trial_hash"],
                candidate_id=row["candidate_id"],
                default=None,
            )
            try:
                artifact_refs = [
                    ArtifactRef.model_validate(item) for item in artifact_items
                ]
            except ValidationError as exc:
                raise self._storage_read_error(
                    "trial_events.artifact_refs_json",
                    row["trial_hash"],
                    row["candidate_id"],
                    exc,
                ) from exc
            try:
                error = (
                    ErrorRecord.model_validate(error_payload)
                    if error_payload is not None
                    else None
                )
                events.append(
                    TrialEvent(
                        trial_hash=row["trial_hash"],
                        event_seq=row["event_seq"],
                        event_id=row["event_id"],
                        candidate_id=row["candidate_id"],
                        event_type=row["event_type"],
                        stage=row["stage"],
                        status=row["status"],
                        event_ts=row["event_ts"],
                        metadata=parsed_metadata,
                        payload=payload,
                        artifact_refs=artifact_refs,
                        error=error,
                    )
                )
            except ValidationError as exc:
                raise self._storage_read_error(
                    "trial_events.row",
                    row["trial_hash"],
                    row["candidate_id"],
                    exc,
                ) from exc
        return events

    def last_event_index(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> int | None:
        """Return the last stored event sequence for the requested scope."""
        with self.manager.get_connection() as conn:
            if candidate_id is None:
                row = conn.execute(
                    "SELECT MAX(event_seq) AS max_seq FROM trial_events WHERE trial_hash = ?",
                    (trial_hash,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT MAX(event_seq) AS max_seq
                    FROM trial_events
                    WHERE trial_hash = ? AND candidate_id = ?
                    """,
                    (trial_hash, candidate_id),
                ).fetchone()
        max_seq = row["max_seq"] if row is not None else None
        return int(max_seq) if max_seq is not None else None

    def has_projection_for_overlay(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        """Return whether a projection-completed event exists for the overlay."""
        with self.manager.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT metadata_json
                FROM trial_events
                WHERE trial_hash = ? AND event_type = ?
                ORDER BY event_seq DESC
                """,
                (trial_hash, TrialEventType.PROJECTION_COMPLETED),
            ).fetchall()

        for row in rows:
            metadata = self._load_json_dict(
                row["metadata_json"],
                label="trial_events.metadata_json",
                trial_hash=trial_hash,
                candidate_id=None,
            )
            try:
                projection_metadata = parse_trial_event_metadata(
                    event_type=TrialEventType.PROJECTION_COMPLETED,
                    metadata=metadata,
                )
            except ValidationError as exc:
                raise self._storage_read_error(
                    "trial_events.metadata_json",
                    trial_hash,
                    None,
                    exc,
                ) from exc
            if (
                isinstance(projection_metadata, ProjectionCompletedEventMetadata)
                and projection_metadata.transform_hash == transform_hash
                and projection_metadata.evaluation_hash == evaluation_hash
            ):
                return True
        return False

    def latest_terminal_event_type(self, trial_hash: str) -> TrialEventType | None:
        """Return the latest terminal trial event without hydrating the full stream."""
        with self.manager.get_connection() as conn:
            row = conn.execute(
                """
                SELECT event_type
                FROM trial_events
                WHERE trial_hash = ? AND event_type IN (?, ?)
                ORDER BY event_seq DESC
                LIMIT 1
                """,
                (
                    trial_hash,
                    TrialEventType.TRIAL_COMPLETED,
                    TrialEventType.TRIAL_FAILED,
                ),
            ).fetchone()
        return TrialEventType(row["event_type"]) if row is not None else None

    def _decode_json_column(
        self,
        raw_value: str | None,
        *,
        label: str,
        trial_hash: str,
        candidate_id: str | None,
        default: object = {},
    ) -> object:
        if raw_value is None:
            return default
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise self._storage_read_error(
                label, trial_hash, candidate_id, exc
            ) from exc

    def _load_json_dict(
        self,
        raw_value: str | None,
        *,
        label: str,
        trial_hash: str,
        candidate_id: str | None,
    ) -> JSONDict:
        decoded = self._decode_json_column(
            raw_value,
            label=label,
            trial_hash=trial_hash,
            candidate_id=candidate_id,
            default={},
        )
        try:
            return _JSON_DICT_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self._storage_read_error(
                label, trial_hash, candidate_id, exc
            ) from exc

    def _load_json_list(
        self,
        raw_value: str | None,
        *,
        label: str,
        trial_hash: str,
        candidate_id: str | None,
    ) -> JSONList:
        decoded = self._decode_json_column(
            raw_value,
            label=label,
            trial_hash=trial_hash,
            candidate_id=candidate_id,
            default=[],
        )
        try:
            return _JSON_LIST_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self._storage_read_error(
                label, trial_hash, candidate_id, exc
            ) from exc

    def _load_json_value(
        self,
        raw_value: str | None,
        *,
        label: str,
        trial_hash: str,
        candidate_id: str | None,
        default: JSONValueType | None,
    ) -> JSONValueType | None:
        decoded = self._decode_json_column(
            raw_value,
            label=label,
            trial_hash=trial_hash,
            candidate_id=candidate_id,
            default=default,
        )
        if decoded is None:
            return None
        try:
            return _JSON_VALUE_ADAPTER.validate_python(decoded)
        except ValidationError as exc:
            raise self._storage_read_error(
                label, trial_hash, candidate_id, exc
            ) from exc

    def _persisted_spec_canonical_hash(
        self,
        spec: SpecBase,
        *,
        row: StorageRow,
    ) -> str:
        persisted_canonical_hash = row["canonical_hash"]
        if isinstance(persisted_canonical_hash, str) and persisted_canonical_hash:
            return persisted_canonical_hash
        try:
            persisted_spec = spec.__class__.model_validate(
                json.loads(row["canonical_json"])
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message="invalid persisted spec payload in specs.canonical_json",
                details={"spec_hash": spec.spec_hash},
            ) from exc
        return persisted_spec.canonical_hash

    def _storage_read_error(
        self,
        label: str,
        trial_hash: str,
        candidate_id: str | None,
        exc: ValidationError | json.JSONDecodeError,
    ) -> StorageError:
        detail = (
            format_validation_error(exc)
            if isinstance(exc, ValidationError)
            else exc.msg
        )
        context = f"trial_hash={trial_hash}"
        if candidate_id is not None:
            context = f"{context}, candidate_id={candidate_id}"
        return StorageError(
            code=ErrorCode.STORAGE_READ,
            message=f"Failed to hydrate {label} ({context}): {detail}",
        )
