from __future__ import annotations

import os
from dataclasses import replace
from statistics import mean, pstdev

from themis.comparison import ComparisonEngine
from themis.comparison.statistics import StatisticalTest
from themis.storage import ExperimentStorage

from common import (
    BASELINE_PROMPT,
    CANDIDATE_PROMPT,
    make_spec,
    register_countdown_extensions,
    run_spec,
)


def metric_by_sample(storage: ExperimentStorage, run_id: str, metric_name: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for rec in storage.load_cached_evaluations(run_id).values():
        if rec.sample_id is None:
            continue
        for score in rec.scores:
            if score.metric_name == metric_name:
                buckets.setdefault(rec.sample_id, []).append(score.value)
    return {
        sample_id: sum(values) / len(values)
        for sample_id, values in buckets.items()
        if values
    }


if __name__ == "__main__":
    register_countdown_extensions()

    dataset_limit = int(os.getenv("COUNTDOWN_LIMIT", "4"))
    repeats = int(os.getenv("COUNTDOWN_REPEATS", "1"))
    if repeats < 1:
        repeats = 1

    base_spec = make_spec(
        run_id="countdown-part8-baseline-r1",
        prompt=BASELINE_PROMPT,
        dataset_limit=dataset_limit,
    )
    base_spec = replace(
        base_spec,
        provider_options={**base_spec.provider_options, "timeout": 20},
    )

    for i in range(1, repeats + 1):
        run_spec(
            replace(base_spec, run_id=f"countdown-part8-baseline-r{i}"),
            workers=1,
            max_retries=1,
            storage_path=".cache/experiments",
            cache=True,
        )
        run_spec(
            replace(
                base_spec,
                prompt=CANDIDATE_PROMPT,
                run_id=f"countdown-part8-candidate-r{i}",
            ),
            workers=1,
            max_retries=1,
            storage_path=".cache/experiments",
            cache=True,
        )

    run_ids = []
    for i in range(1, repeats + 1):
        run_ids.append(f"countdown-part8-baseline-r{i}")
        run_ids.append(f"countdown-part8-candidate-r{i}")

    engine_t = ComparisonEngine(
        storage_path=".cache/experiments",
        statistical_test=StatisticalTest.T_TEST,
        alpha=0.05,
        multiple_comparison_correction="holm-bonferroni",
    )
    report_t = engine_t.compare_runs(run_ids, metrics=["CountdownValidity"])
    for row in report_t.pairwise_results[:2]:
        p_raw = row.test_result.p_value if row.test_result else None
        print("t_test", row.run_a_id, row.run_b_id, row.delta, p_raw, row.corrected_p_value)

    engine_b = ComparisonEngine(
        storage_path=".cache/experiments",
        statistical_test=StatisticalTest.BOOTSTRAP,
        alpha=0.05,
        n_bootstrap=1000,
    )
    report_b = engine_b.compare_runs(
        ["countdown-part8-baseline-r1", "countdown-part8-candidate-r1"],
        metrics=["CountdownValidity"],
    )
    row_b = report_b.pairwise_results[0]
    ci = row_b.test_result.confidence_interval if row_b.test_result else None
    print("bootstrap_mode", row_b.test_result.inference_mode if row_b.test_result else None)
    print("bootstrap_ci", ci)

    storage = ExperimentStorage(".cache/experiments")
    a = metric_by_sample(storage, "countdown-part8-baseline-r1", "CountdownValidity")
    b = metric_by_sample(storage, "countdown-part8-candidate-r1", "CountdownValidity")

    common = sorted(set(a) & set(b))
    deltas = [a[sid] - b[sid] for sid in common]
    hard_disagreements = [sid for sid in common if abs(a[sid] - b[sid]) >= 1.0]

    print("n_common", len(common))
    print("mean_delta", mean(deltas) if deltas else 0.0)
    print("delta_std", pstdev(deltas) if len(deltas) > 1 else 0.0)
    print("hard_disagreement_rate", len(hard_disagreements) / len(common) if common else 0.0)
    print("dataset_limit", dataset_limit)
    print("repeats", repeats)
