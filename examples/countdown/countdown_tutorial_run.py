from __future__ import annotations

from common import (
    BASELINE_PROMPT,
    DEFAULT_STORAGE,
    make_spec,
    register_countdown_extensions,
    run_spec,
)


if __name__ == "__main__":
    register_countdown_extensions()

    spec = make_spec(
        run_id="countdown-localchat-run",
        prompt=BASELINE_PROMPT,
        dataset_limit=20,
    )
    report = run_spec(
        spec, workers=1, max_retries=3, storage_path=DEFAULT_STORAGE, cache=True
    )

    solve_rate = report.evaluation_report.metrics["CountdownValidity"].mean
    print(f"run_id={spec.run_id}")
    print(f"solve_rate={solve_rate:.4f}")
