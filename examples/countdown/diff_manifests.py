from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def _value_at(payload: Mapping, dotted_key: str):
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def manifest_diff(manifest_a: Mapping, manifest_b: Mapping) -> dict[str, dict]:
    keys = [
        "model.identifier",
        "model.provider",
        "sampling.temperature",
        "sampling.top_p",
        "sampling.max_tokens",
        "num_samples",
        "dataset.fingerprint",
        "dataset.benchmark_id",
        "prompt.template_hash",
        "evaluation.extractor",
        "evaluation.metrics",
        "seeds.provider_seed",
        "seeds.sampling_seed",
        "git_commit_hash",
    ]
    out = {}
    for key in keys:
        a_val = _value_at(manifest_a, key)
        b_val = _value_at(manifest_b, key)
        if a_val != b_val:
            out[key] = {"a": a_val, "b": b_val}
    return out


if __name__ == "__main__":
    run_a = "countdown-part6-backends-smoke"
    run_b = "countdown-part7-ops-smoke"

    in_path = Path("outputs/countdown_part9/manifest_index.json")
    if not in_path.exists():
        raise SystemExit("Missing manifest_index.json. Run build_manifest_index.py first.")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", {})

    if run_a not in runs or run_b not in runs:
        raise SystemExit("manifest_index missing expected run IDs")

    diffs = manifest_diff(runs[run_a]["manifest"], runs[run_b]["manifest"])

    out_path = Path("outputs/countdown_part9/manifest_diff.json")
    out_path.write_text(
        json.dumps({"run_a": run_a, "run_b": run_b, "diffs": diffs}, indent=2),
        encoding="utf-8",
    )

    print("diff_count", len(diffs))
    print("manifest_diff", out_path)
