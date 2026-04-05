"""Filesystem-backed JSONL run store."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from themis.core.base import JSONValue
from themis.core.events import RunEvent, event_from_dict
from themis.core.snapshot import RunSnapshot, StoredRun, snapshot_from_dict
from themis.core.stores.base import ProjectionRefreshingStore


class JsonlRunStore(ProjectionRefreshingStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def initialize(self) -> None:
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        run_root = self._run_root(snapshot.run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "snapshot.json").write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._bootstrap_projections(snapshot)

    def persist_event(self, event: RunEvent) -> None:
        run_root = self._run_root(event.run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        with (run_root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n"
            )
        snapshot = self._load_snapshot(event.run_id)
        if snapshot is not None:
            self._refresh_projections_for_event(snapshot, event)

    def query_events(self, run_id: str) -> list[RunEvent]:
        events_path = self._run_root(run_id) / "events.jsonl"
        if not events_path.is_file():
            return []
        events: list[RunEvent] = []
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                try:
                    events.append(event_from_dict(payload))
                except KeyError:
                    continue
        return events

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        projection_path = (
            self._run_root(run_id) / "projections" / f"{projection_name}.json"
        )
        if not projection_path.is_file():
            return None
        return json.loads(projection_path.read_text(encoding="utf-8"))

    def store_blob(self, blob: bytes, media_type: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        ref = f"sha256:{digest}"
        blob_path = self.root / "blobs" / f"{digest}.blob"
        meta_path = self.root / "blobs" / f"{digest}.meta.json"
        if not blob_path.exists():
            blob_path.write_bytes(blob)
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps({"media_type": media_type}, sort_keys=True), encoding="utf-8"
            )
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        digest = blob_ref.removeprefix("sha256:")
        blob_path = self.root / "blobs" / f"{digest}.blob"
        meta_path = self.root / "blobs" / f"{digest}.meta.json"
        if not blob_path.is_file() or not meta_path.is_file():
            return None
        media_type = json.loads(meta_path.read_text(encoding="utf-8"))["media_type"]
        return media_type, blob_path.read_bytes()

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            return None
        return StoredRun(snapshot=snapshot, events=self.query_events(run_id))

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        snapshot_path = self._run_root(run_id) / "snapshot.json"
        if not snapshot_path.is_file():
            return None
        return snapshot_from_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        projections_root = self._run_root(run_id) / "projections"
        projections_root.mkdir(parents=True, exist_ok=True)
        (projections_root / f"{projection_name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _run_root(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        path = self.root / "stage_cache" / stage_name / f"{cache_key}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        path = self.root / "stage_cache" / stage_name / f"{cache_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def clear_run(self, run_id: str) -> None:
        shutil.rmtree(self._run_root(run_id), ignore_errors=True)


def jsonl_store(root: str | Path) -> JsonlRunStore:
    return JsonlRunStore(root)
