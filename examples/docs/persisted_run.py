from __future__ import annotations

from pathlib import Path

from themis import Experiment, Reporter, get_execution_state, sqlite_store
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def run_example(root: Path) -> dict[str, object]:
    """Run against SQLite, inspect the stored state, and export a report."""

    store_path = root / "runs" / "themis.sqlite3"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite_store(store_path)
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="sqlite", parameters={"path": str(store_path)}),
        datasets=[
            Dataset(
                dataset_id="sample",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )

    result = experiment.run(store=store)
    state = get_execution_state(store, result.run_id)
    report = Reporter(store).export_markdown(result.run_id)
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "store_path": str(store_path),
        "state_status": state.status.value,
        "report_preview": report.splitlines()[:4],
    }


if __name__ == "__main__":
    print(run_example(Path(".")))
