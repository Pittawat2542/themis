from __future__ import annotations

from themis.storage import ExperimentStorage


if __name__ == "__main__":
    storage = ExperimentStorage(".cache/experiments")
    run_id = "countdown-candidate-v1"

    run_index = {run.run_id: run for run in storage.list_runs(limit=500)}
    if run_id not in run_index:
        raise SystemExit(f"run not found: {run_id}")

    snapshot = run_index[run_id].config_snapshot or {}
    manifest = snapshot.get("reproducibility_manifest", {})
    manifest_hash = snapshot.get("manifest_hash")

    assert manifest_hash, "missing manifest_hash"
    assert manifest.get("dataset", {}).get("fingerprint"), "missing dataset fingerprint"
    assert manifest.get("prompt", {}).get("template_hash"), "missing prompt hash"
    assert manifest.get("evaluation"), "missing evaluation fingerprint"

    print("run_id", run_id)
    print("manifest_hash", manifest_hash)
    print("dataset_fingerprint", manifest["dataset"]["fingerprint"])
    print("prompt_hash", manifest["prompt"]["template_hash"])
