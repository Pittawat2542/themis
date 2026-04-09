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
    """Persist runs in MongoDB while storing blobs on the filesystem."""

    def __init__(self, url: str, database: str, blob_root: str | Path) -> None:
        self.url = url
        self.database = database
        self.blob_root = Path(blob_root)
        self._database_handle = None

    def initialize(self) -> None:
        self.blob_root.mkdir(parents=True, exist_ok=True)
        database = self._db()
        database["run_events"].create_index(
            [("run_id", 1), ("sequence", 1)], unique=True
        )
        database["run_event_counters"].create_index([("run_id", 1)], unique=True)

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        self._db()["run_snapshots"].replace_one(
            {"run_id": snapshot.run_id},
            {
                "run_id": snapshot.run_id,
                "snapshot_json": snapshot.model_dump(mode="json"),
            },
            upsert=True,
        )
        self._bootstrap_projections(snapshot)

    def persist_event(self, event: RunEvent) -> None:
        sequence = self._allocate_sequence(event.run_id)
        self._db()["run_events"].insert_one(
            {
                "run_id": event.run_id,
                "sequence": sequence,
                "event_type": event.event_type,
                "event_json": event.model_dump(mode="json"),
            }
        )
        snapshot = self._load_snapshot(event.run_id)
        if snapshot is not None:
            self._refresh_projections_for_event(snapshot, event)

    def query_events(self, run_id: str) -> list[RunEvent]:
        rows = sorted(
            self._db()["run_events"].find({"run_id": run_id}),
            key=lambda row: row["sequence"],
        )
        events: list[RunEvent] = []
        for row in rows:
            try:
                events.append(event_from_dict(dict(row["event_json"])))
            except KeyError:
                continue
        return events

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        row = self._db()["run_projections"].find_one(
            {"run_id": run_id, "projection_name": projection_name}
        )
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
            meta_path.write_text(
                json.dumps({"media_type": media_type}, sort_keys=True), encoding="utf-8"
            )
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        digest = blob_ref.removeprefix("sha256:")
        blob_path = self.blob_root / f"{digest}.blob"
        meta_path = self.blob_root / f"{digest}.meta.json"
        if not blob_path.is_file() or not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))[
            "media_type"
        ], blob_path.read_bytes()

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            return None
        return StoredRun(snapshot=snapshot, events=self.query_events(run_id))

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        row = self._db()["run_snapshots"].find_one({"run_id": run_id})
        if row is None:
            return None
        return snapshot_from_dict(dict(row["snapshot_json"]))

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        self._db()["run_projections"].replace_one(
            {"run_id": run_id, "projection_name": projection_name},
            {
                "run_id": run_id,
                "projection_name": projection_name,
                "projection_json": payload,
            },
            upsert=True,
        )

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        row = self._db()["stage_cache"].find_one(
            {"stage_name": stage_name, "cache_key": cache_key}
        )
        if row is None:
            return None
        return row["payload_json"]

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        self._db()["stage_cache"].replace_one(
            {"stage_name": stage_name, "cache_key": cache_key},
            {
                "stage_name": stage_name,
                "cache_key": cache_key,
                "payload_json": payload,
            },
            upsert=True,
        )

    def clear_run(self, run_id: str) -> None:
        self._db()["run_events"].delete_many({"run_id": run_id})
        self._db()["run_event_counters"].delete_many({"run_id": run_id})
        self._db()["run_projections"].delete_many({"run_id": run_id})
        self._db()["run_snapshots"].delete_many({"run_id": run_id})

    def _allocate_sequence(self, run_id: str) -> int:
        row = self._db()["run_event_counters"].find_one_and_update(
            {"run_id": run_id},
            {"$inc": {"next_sequence": 1}},
            upsert=True,
            return_document="after",
        )
        if row is None:
            raise RuntimeError(
                f"Unable to allocate MongoDB event sequence for {run_id}"
            )
        return int(row["next_sequence"]) - 1

    def _db(self):
        if self._database_handle is not None:
            return self._database_handle
        try:
            pymongo = importlib.import_module("pymongo")
        except ImportError as exc:
            raise ImportError(
                "MongoDB support requires the optional 'mongodb' dependency."
            ) from exc
        client = pymongo.MongoClient(self.url)
        self._database_handle = client[self.database]
        return self._database_handle


def mongodb_store(url: str, database: str, blob_root: str | Path) -> MongoDbRunStore:
    """Create a MongoDB-backed run store."""

    return MongoDbRunStore(url, database, blob_root)
