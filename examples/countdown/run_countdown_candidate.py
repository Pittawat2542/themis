from __future__ import annotations

from common import (
    CANDIDATE_PROMPT,
    DEFAULT_STORAGE,
    make_spec,
    register_countdown_extensions,
    run_spec,
)


if __name__ == "__main__":
    register_countdown_extensions()

    spec = make_spec(
        run_id="countdown-candidate-v1",
        prompt=CANDIDATE_PROMPT,
        dataset_limit=50,
        provider_seed=42,
    )
    report = run_spec(spec, workers=1, max_retries=3, storage_path=DEFAULT_STORAGE, cache=True)

    print(f"run_id={spec.run_id}")
    print(f"countdown_validity_mean={report.evaluation_report.metrics['CountdownValidity'].mean:.4f}")
