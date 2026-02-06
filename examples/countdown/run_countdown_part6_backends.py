from __future__ import annotations

from pathlib import Path

from themis.backends.execution import LocalExecutionBackend
from themis.backends.storage import LocalFileStorageBackend
from themis.experiment import export as export_utils
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, StorageSpec

from common import (
    BASELINE_PROMPT,
    DEFAULT_STORAGE,
    make_spec,
    register_countdown_extensions,
)


if __name__ == "__main__":
    register_countdown_extensions()

    spec = make_spec(
        run_id="countdown-part6-backends",
        prompt=BASELINE_PROMPT,
        dataset_limit=30,
    )

    execution_backend = LocalExecutionBackend(max_workers=2)
    storage_backend = LocalFileStorageBackend(DEFAULT_STORAGE)

    try:
        report = ExperimentSession().run(
            spec,
            execution=ExecutionSpec(backend=execution_backend, workers=2, max_retries=3),
            storage=StorageSpec(backend=storage_backend, cache=True),
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
        run_id=spec.run_id,
    )

    print("run_id", spec.run_id)
    print("countdown_validity_mean", report.evaluation_report.metrics["CountdownValidity"].mean)
    print("bundle_json", paths["json"])
