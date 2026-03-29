"""MongoDB-backed run store with filesystem blob storage."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from themis.core.base import JSONValue
from themis.core.events import RunEvent, event_from_dict
from themis.core.snapshot import RunSnapshot, StoredRun, snapshot_from_dict
from themis.core.stores.base import ProjectionRefreshingStore


class MongoDbRunStore(ProjectionRefreshingStore):
    def __init__(self, url: str, database: str, blob_root: str | Path) -> None:
        self.url = url
        self.database = database
        self.blob_root = Path(blob_root)
        self._database_handle = None

    def initialize(self) -> None:
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self._db()

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        self._db()["run_snapshots"].replace_one(
            {"run_id": snapshot.run_id},
            {"run_id": snapshot.run_id, "snapshot_json": snapshot.model_dump(mode="json")},
            upsert=True,
        )
        self._refresh_projections(snapshot.run_id)

    def persist_event(self, event: RunEvent) -> None:
        sequence = len(self.query_events(event.run_id))
        self._db()["run_events"].insert_one(
            {
                "run_id": event.run_id,
                "sequence": sequence,
                "event_type": event.event_type,
                "event_json": event.model_dump(mode="json"),
            }
        )
        self._refresh_projections(event.run_id)

    def query_events(self, run_id: str) -> list[RunEvent]:
        rows = sorted(self._db()["run_events"].find({"run_id": run_id}), key=lambda row: row["sequence"])
        events: list[RunEvent] = []
        for row in rows:
            try:
                events.append(event_from_dict(dict(row["event_json"])))
            except KeyError:
                continue
        return events

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        row = self._db()["run_projections"].find_one({"run_id": run_id, "projection_name": projection_name})
        if row is None:
            return None
        return row["projection_json"]

    def store_blob(self, blob: bytes, media_type: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        ref = f"sha256:{digest}"
        blob_path = self.blob_root / f"{digest}.blob"
        meta_path = self.blob_root / f"{digest}.meta.json"
        if not blob_path.exists():
            blob_path.write_bytes(blob)
        if not meta_path.exists():
            meta_path.write_text(json.dumps({"media_type": media_type}, sort_keys=True), encoding="utf-8")
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        digest = blob_ref.removeprefix("sha256:")
        blob_path = self.blob_root / f"{digest}.blob"
        meta_path = self.blob_root / f"{digest}.meta.json"
        if not blob_path.is_file() or not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))["media_type"], blob_path.read_bytes()

    def resume(self, run_id: str) -> StoredRun | None:
        row = self._db()["run_snapshots"].find_one({"run_id": run_id})
        if row is None:
            return None
        return StoredRun(
            snapshot=snapshot_from_dict(dict(row["snapshot_json"])),
            events=self.query_events(run_id),
        )

    def _write_projection(self, run_id: str, projection_name: str, payload: JSONValue) -> None:
        self._db()["run_projections"].replace_one(
            {"run_id": run_id, "projection_name": projection_name},
            {
                "run_id": run_id,
                "projection_name": projection_name,
                "projection_json": payload,
            },
            upsert=True,
        )

    def _db(self):
        if self._database_handle is not None:
            return self._database_handle
        try:
            pymongo = importlib.import_module("pymongo")
        except ImportError as exc:
            raise ImportError("MongoDB support requires the optional 'mongodb' dependency.") from exc
        client = pymongo.MongoClient(self.url)
        self._database_handle = client[self.database]
        return self._database_handle


def mongodb_store(url: str, database: str, blob_root: str | Path) -> MongoDbRunStore:
    return MongoDbRunStore(url, database, blob_root)
