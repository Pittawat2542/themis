from __future__ import annotations

from pathlib import Path

import themis
from themis.backends.execution import LocalExecutionBackend
from themis.storage import ExperimentStorage
from themis.experiment import export as export_utils

from common import (
    BASELINE_PROMPT,
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    DEFAULT_STORAGE,
    load_countdown_for_themis,
    register_countdown_extensions,
)


if __name__ == "__main__":
    register_countdown_extensions()

    execution_backend = LocalExecutionBackend(max_workers=2)
    storage_backend = ExperimentStorage(DEFAULT_STORAGE)

    try:
        report = themis.evaluate(
            load_countdown_for_themis(limit=30, split="train"),
            model=DEFAULT_MODEL,
            prompt=BASELINE_PROMPT,
            metrics=["countdown_validity"],
            provider_options={"api_base": DEFAULT_API_BASE},
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            workers=2,
            max_retries=3,
            run_id="countdown-part6-backends",
            execution_backend=execution_backend,
            storage_backend=storage_backend,
            resume=True,
        )
    finally:
        execution_backend.shutdown()

    bundle_dir = Path("outputs/countdown_part6")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    paths = export_utils.export_report_bundle(
        report,
        csv_path=bundle_dir / "report.csv",
        json_path=bundle_dir / "report.json",
        html_path=bundle_dir / "report.html",
        summary_path=bundle_dir / "summary.json",
        run_id="countdown-part6-backends",
    )

    print("run_id countdown-part6-backends")
    print(
        "countdown_validity_mean",
        report.evaluation_report.metrics["CountdownValidity"].mean,
    )
    print("bundle_json", paths["json"])
