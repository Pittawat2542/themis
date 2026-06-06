from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import cast

from themis import InMemoryRunStore
from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.dataset_inputs import dataset_from_inline, dataset_from_jsonl
from themis.core.experiment import Experiment
from themis.core.models import Dataset
from tests.cli.helpers import run_cli


def _run_python_dataset(dataset: Dataset) -> tuple[str, dict[str, JSONValue]]:
    store = InMemoryRunStore()
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[dataset],
    )
    result = experiment.run(store=store)
    benchmark = cast(
        dict[str, JSONValue],
        store.get_projection(result.run_id, "benchmark_result"),
    )
    return result.run_id, benchmark


def test_quick_eval_inline_matches_python_api() -> None:
    dataset = dataset_from_inline(
        input_value={"question": "2+2"}, expected_output={"answer": "4"}
    )
    python_run_id, benchmark = _run_python_dataset(dataset)

    cli_result = run_cli(
        "quick-eval",
        "inline",
        "--input-json",
        '{"question":"2+2"}',
        "--expected-output-json",
        '{"answer":"4"}',
    )

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["run_id"] == python_run_id
    assert payload["metric_means"] == benchmark["metric_means"]


def test_quick_eval_file_matches_python_api(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n'
    )

    dataset = dataset_from_jsonl(path)
    python_run_id, benchmark = _run_python_dataset(dataset)

    cli_result = run_cli("quick-eval", "file", "--path", str(path))

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["run_id"] == python_run_id
    assert payload["metric_means"] == benchmark["metric_means"]


def test_quick_eval_huggingface_reports_missing_dependency() -> None:
    if importlib.util.find_spec("datasets") is not None:
        return

    cli_result = run_cli(
        "quick-eval",
        "huggingface",
        "--dataset",
        "demo",
        "--split",
        "train",
        "--input-field",
        "prompt",
        "--expected-output-field",
        "answer",
    )

    assert cli_result.returncode != 0
    assert 'uv add "themis-eval[datasets]"' in cli_result.stderr


def test_quick_eval_huggingface_uses_optional_datasets_module(tmp_path: Path) -> None:
    package_root = tmp_path / "fakepkgs" / "datasets"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        """
def load_dataset(dataset_name, *, split):
    assert dataset_name == "demo"
    assert split == "train"
    return [
        {"id": "row-1", "prompt": {"question": "2+2"}, "answer": {"answer": "4"}},
    ]
""".strip()
    )

    cli_result = run_cli(
        "quick-eval",
        "huggingface",
        "--dataset",
        "demo",
        "--split",
        "train",
        "--input-field",
        "prompt",
        "--expected-output-field",
        "answer",
        "--case-id-field",
        "id",
        env={
            "PYTHONPATH": f"{tmp_path / 'fakepkgs'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        },
    )

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["status"] == "completed"
    assert payload["metric_means"] == {"builtin/exact_match": 1.0}


def test_quick_eval_benchmark_delegates_to_catalog_run(tmp_path: Path) -> None:
    package_root = tmp_path / "fakepkgs" / "datasets"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(
        """
def load_dataset(dataset_name, *args, split=None, revision=None, **kwargs):
    del args, revision, kwargs
    assert dataset_name == "TIGER-Lab/MMLU-Pro"
    assert split == "test"
    return [
        {
            "item_id": "mmlu-pro-1",
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Mercury"],
            "answer": "B",
            "category": "astronomy",
            "src": "fixture",
        }
    ]
""".strip()
    )

    cli_result = run_cli(
        "quick-eval",
        "benchmark",
        "--name",
        "mmlu_pro",
        env={
            "PYTHONPATH": f"{package_root.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        },
    )

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["status"] == "completed"
    assert payload["metric_means"] == {"builtin/choice_accuracy": 1.0}
