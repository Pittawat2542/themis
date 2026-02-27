from __future__ import annotations

from themis.comparison import compare_runs
from themis.comparison import StatisticalTest


if __name__ == "__main__":
    baseline = "countdown-baseline-v1"
    candidate = "countdown-candidate-v1"
    min_delta = 0.0

    cmp = compare_runs(
        run_ids=[baseline, candidate],
        storage_path=".cache/experiments",
        metrics=["CountdownValidity"],
        statistical_test=StatisticalTest.BOOTSTRAP,
        alpha=0.05,
    )

    row = cmp.pairwise_results[0]
    delta_candidate_vs_baseline = row.run_b_mean - row.run_a_mean

    print("baseline_mean", row.run_a_mean)
    print("candidate_mean", row.run_b_mean)
    print("candidate_delta", delta_candidate_vs_baseline)

    if delta_candidate_vs_baseline < min_delta:
        raise SystemExit("Quality gate failed: candidate underperforms baseline")

    print("gate", "pass")
