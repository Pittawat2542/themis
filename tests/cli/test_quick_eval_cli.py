from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from themis import InMemoryRunStore, evaluate
from themis.core.base import JSONValue
from themis.core.config import StorageConfig
from themis.core.dataset_inputs import dataset_from_inline, dataset_from_jsonl


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "themis.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


def test_quick_eval_inline_matches_python_api() -> None:
    store = InMemoryRunStore()
    dataset = dataset_from_inline(input_value={"question": "2+2"}, expected_output={"answer": "4"})
    python_result = evaluate(
        model="builtin/demo_generator",
        data=[dataset],
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
        store=store,
    )
    benchmark = cast(dict[str, JSONValue], store.get_projection(python_result.run_id, "benchmark_result"))

    cli_result = _run_cli(
        "quick-eval",
        "inline",
        "--input-json",
        '{"question":"2+2"}',
        "--expected-output-json",
        '{"answer":"4"}',
    )

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["run_id"] == python_result.run_id
    assert payload["metric_means"] == benchmark["metric_means"]


def test_quick_eval_file_matches_python_api(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n')

    store = InMemoryRunStore()
    dataset = dataset_from_jsonl(path)
    python_result = evaluate(
        model="builtin/demo_generator",
        data=[dataset],
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
        store=store,
    )
    benchmark = cast(dict[str, JSONValue], store.get_projection(python_result.run_id, "benchmark_result"))

    cli_result = _run_cli("quick-eval", "file", "--path", str(path))

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["run_id"] == python_result.run_id
    assert payload["metric_means"] == benchmark["metric_means"]


def test_quick_eval_huggingface_reports_missing_dependency() -> None:
    cli_result = _run_cli(
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
    assert "pip install themis-eval[datasets]" in cli_result.stderr


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

    cli_result = _run_cli(
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
        env={"PYTHONPATH": f"{tmp_path / 'fakepkgs'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"},
    )

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["status"] == "completed"
    assert payload["metric_means"] == {"builtin/exact_match": 1.0}


def test_quick_eval_benchmark_delegates_to_catalog_run() -> None:
    cli_result = _run_cli("quick-eval", "benchmark", "--name", "mmlu_pro")

    assert cli_result.returncode == 0, cli_result.stderr
    payload = json.loads(cli_result.stdout)
    assert payload["status"] == "completed"
    assert payload["metric_means"] == {"builtin/exact_match": 1.0}
