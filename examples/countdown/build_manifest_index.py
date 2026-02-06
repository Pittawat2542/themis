from __future__ import annotations

import json
from pathlib import Path

from themis.experiment.manifest import manifest_hash, validate_reproducibility_manifest
from themis.storage import ExperimentStorage


if __name__ == "__main__":
    run_ids = [
        "countdown-part6-backends-smoke",
        "countdown-part7-ops-smoke",
    ]

    storage = ExperimentStorage(".cache/experiments")
    run_index = {run.run_id: run for run in storage.list_runs(limit=500)}

    manifests = {}
    for run_id in run_ids:
        if run_id not in run_index:
            raise SystemExit(f"run not found: {run_id}")
        snapshot = run_index[run_id].config_snapshot or {}
        manifest = snapshot.get("reproducibility_manifest")
        if not isinstance(manifest, dict):
            raise SystemExit(f"missing reproducibility_manifest for run: {run_id}")
        validate_reproducibility_manifest(manifest)
        manifests[run_id] = {
            "manifest": manifest,
            "manifest_hash": manifest_hash(manifest),
            "status": run_index[run_id].status.value,
        }

    out_dir = Path("outputs/countdown_part9")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest_index.json"
    out_path.write_text(json.dumps({"runs": manifests}, indent=2), encoding="utf-8")
    print("manifest_index", out_path)
