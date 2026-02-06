from __future__ import annotations

import json
from pathlib import Path


if __name__ == "__main__":
    allowed = {
        "prompt.template_hash",
        "git_commit_hash",
    }

    in_path = Path("outputs/countdown_part9/manifest_diff.json")
    if not in_path.exists():
        raise SystemExit("Missing manifest_diff.json. Run diff_manifests.py first.")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    diffs = payload.get("diffs", {})

    forbidden = {k: v for k, v in diffs.items() if k not in allowed}
    out_path = Path("outputs/countdown_part9/reproducibility_audit.json")
    out_path.write_text(
        json.dumps(
            {
                "runs": [payload.get("run_a"), payload.get("run_b")],
                "allowed_drift": sorted(allowed),
                "diffs": diffs,
                "forbidden_drift": forbidden,
                "pass": not bool(forbidden),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("forbidden_count", len(forbidden))
    print("audit", out_path)

    if forbidden:
        raise SystemExit("Reproducibility gate failed: found forbidden manifest drift")

    print("gate", "pass")
