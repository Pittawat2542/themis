from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from themis.core.experiment import Experiment
from themis.core.results import RunResult
from themis.core.store import RunStore
from themis.core.stores.factory import create_run_store
from themis.launcher import load_core_experiment


@pytest.fixture
def write_experiment_config(tmp_path: Path) -> Callable[..., Path]:
    """Write a minimal deterministic experiment config for CLI and store tests."""

    def _write(
        name: str = "experiment.yaml",
        *,
        answer: str = "4",
        seed: int | None = 7,
        store_path: Path | None = None,
        queue_root: Path | None = None,
        batch_root: Path | None = None,
    ) -> Path:
        config_path = tmp_path / name
        resolved_store_path = store_path or (tmp_path / "runs.sqlite3")
        resolved_queue_root = queue_root or (tmp_path / "queue")
        resolved_batch_root = batch_root or (tmp_path / "batch")
        definition_module = f"{config_path.stem}_definition"
        seeds = [] if seed is None else [seed]
        (tmp_path / f"{definition_module}.py").write_text(
            f'''from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(
        dataset_id="cases",
        cases=[Case(
            case_id="case-1",
            input={{"question": "2+2"}},
            expected_output={{"answer": "{answer}"}},
        )],
    )],
    generation=Generation(
        generator="builtin/demo_generator",
        reducer="builtin/majority_vote",
    ),
    evaluation=Evaluation(
        metrics=["builtin/exact_match"],
        parser="builtin/json_identity",
    ),
    seeds={seeds!r},
)
''',
            encoding="utf-8",
        )
        config_path.write_text(
            f"""
definition: {definition_module}:experiment
storage:
  target: sqlite
  kwargs:
    path: {resolved_store_path}
runtime:
  queue_root: {resolved_queue_root}
  batch_root: {resolved_batch_root}
""".strip(),
            encoding="utf-8",
        )
        return config_path

    return _write


@pytest.fixture
def run_config_experiment() -> Callable[[Path], tuple[Experiment, RunStore, RunResult]]:
    """Compile and execute an experiment config into its configured store."""

    def _run(config_path: Path) -> tuple[Experiment, RunStore, RunResult]:
        experiment = load_core_experiment(config_path)
        store = create_run_store(experiment.storage)
        store.initialize()
        result = experiment.run(store=store)
        return experiment, store, result

    return _run
