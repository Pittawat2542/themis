from __future__ import annotations

from pathlib import Path

import pytest

from themis.launcher import load_core_experiment


def _write_definition(root: Path) -> None:
    (root / "definition.py").write_text(
        """from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(
        dataset_id="sample",
        cases=[Case(case_id="case-1", input="4", expected_output="4")],
    )],
    generation=Generation(generator="builtin/demo_generator"),
    evaluation=Evaluation(
        metrics=["builtin/exact_match"],
        parser="builtin/json_identity",
    ),
)
""",
        encoding="utf-8",
    )


def test_launcher_imports_python_experiment_and_resolves_paths(tmp_path: Path) -> None:
    _write_definition(tmp_path)
    launcher = tmp_path / "experiment.yaml"
    launcher.write_text(
        """definition: definition:experiment
storage:
  target: sqlite
  kwargs:
    path: runs.sqlite3
runtime:
  max_concurrency: 4
""",
        encoding="utf-8",
    )

    experiment = load_core_experiment(launcher)

    assert experiment.storage.kwargs["path"] == str(tmp_path / "runs.sqlite3")
    assert experiment.runtime.max_concurrent_tasks == 4
    assert experiment.datasets[0].dataset_id == "sample"


def test_launcher_accepts_zero_argument_factory(tmp_path: Path) -> None:
    _write_definition(tmp_path)
    definition = tmp_path / "definition.py"
    definition.write_text(
        definition.read_text(encoding="utf-8")
        + "\ndef build():\n    return experiment\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "experiment.toml"
    launcher.write_text('definition = "definition:build"\n', encoding="utf-8")

    assert load_core_experiment(launcher).datasets[0].dataset_id == "sample"


def test_launcher_rejects_v4_config_shape(tmp_path: Path) -> None:
    launcher = tmp_path / "experiment.yaml"
    launcher.write_text("generation:\n  generator: builtin/demo_generator\n")

    with pytest.raises(ValueError, match="v5 does not define experiments"):
        load_core_experiment(launcher)
