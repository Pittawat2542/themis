from __future__ import annotations

import json
from pathlib import Path

from themis.experiment.comparison import compare_runs
from themis.experiment.comparison import StatisticalTest
from themis.experiment import export as export_utils

from common import (
    CANDIDATE_PROMPT,
    make_spec,
    register_countdown_extensions,
    run_spec,
)


if __name__ == "__main__":
    register_countdown_extensions()

    baseline = "countdown-baseline-v1"
    candidate = "countdown-candidate-v1"

    # Re-run candidate with same run_id to reuse cache and rebuild ExperimentReport object.
    spec = make_spec(
        run_id=candidate,
        prompt=CANDIDATE_PROMPT,
        dataset_limit=50,
        provider_seed=42,
    )
    report = run_spec(
        spec, workers=1, max_retries=3, storage_path=".cache/experiments", cache=True
    )

    cmp = compare_runs(
        run_ids=[baseline, candidate],
        storage_path=".cache/experiments",
        metrics=["CountdownValidity"],
        statistical_test=StatisticalTest.BOOTSTRAP,
        alpha=0.05,
    )
    row = cmp.pairwise_results[0]

    bundle_dir = Path("outputs/countdown_bundle_v1")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    outputs = export_utils.export_report_bundle(
        report,
        csv_path=bundle_dir / "report.csv",
        json_path=bundle_dir / "report.json",
        html_path=bundle_dir / "report.html",
        summary_path=bundle_dir / "summary.json",
        run_id=candidate,
    )

    index_payload = {
        "baseline_run_id": baseline,
        "candidate_run_id": candidate,
        "metric": "CountdownValidity",
        "baseline_mean": row.run_a_mean,
        "candidate_mean": row.run_b_mean,
        "candidate_delta": row.run_b_mean - row.run_a_mean,
        "significant": row.is_significant() if row.test_result else None,
        "p_value": row.test_result.p_value if row.test_result else None,
        "manifest_hash": report.metadata.get("manifest_hash"),
        "artifacts": {name: str(path) for name, path in outputs.items()},
    }

    index_path = bundle_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    print("bundle_index", index_path)
