from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import themis.catalog as catalog
from themis.catalog.benchmarks.codeforces.metric import SandboxExecutionResult
from themis.catalog.benchmarks.humaneval import dataset as humaneval_dataset
from themis.catalog.benchmarks.humaneval.dataset import (
    BuiltinHumanEvalDatasetProvider,
)
from themis.catalog.benchmarks.humaneval.metric import HumanEvalExecutionMetric
from themis.catalog.common import summarize_humaneval, summarize_humaneval_plus
from themis.orchestration.trial_planner import TrialPlanner
from themis.types.events import ScoreRow


def _humaneval_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": "HumanEval/0",
        "entry_point": "add",
        "prompt": (
            "def add(a: int, b: int) -> int:\n"
            '    """Return the sum of two integers."""\n'
        ),
        "canonical_solution": "    return a + b\n",
        "base_input": [[1, 2], [4, 5]],
        "plus_input": [[10, 20]],
        "atol": 0.0,
    }
    row.update(overrides)
    return row


class _FakeExecutor:
    def __init__(self, results: list[SandboxExecutionResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        *,
        code: str,
        language: str,
        stdin: str = "",
        files: dict[str, str] | None = None,
        args: list[str] | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: float | None = None,
    ) -> SandboxExecutionResult:
        self.calls.append(
            {
                "code": code,
                "language": language,
                "stdin": stdin,
                "files": dict(files or {}),
                "args": list(args or []),
                "timeout_seconds": timeout_seconds,
                "memory_limit_mb": memory_limit_mb,
            }
        )
        return self._results.pop(0)


class _StubProjectionRepo:
    def __init__(self, rows: list[ScoreRow]) -> None:
        self._rows = list(rows)

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: list[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del evaluation_hash
        allowed = set(trial_hashes or [])
        for row in self._rows:
            if allowed and row.trial_hash not in allowed:
                continue
            if metric_id is not None and row.metric_id != metric_id:
                continue
            yield row


class _StubSummary:
    def __init__(self, trial_hash: str, item_id: str) -> None:
        self.trial_hash = trial_hash
        self.model_id = "demo-model"
        self.task_id = None
        self.benchmark_id = "humaneval"
        self.slice_id = "humaneval"
        self.prompt_variant_id = "humaneval-default"
        self.dimensions: dict[str, str] = {}
        self.item_id = item_id
        self.status = "ok"


class _StubResult:
    def __init__(self, rows: list[ScoreRow], item_ids_by_trial: dict[str, str]) -> None:
        self.projection_repo = _StubProjectionRepo(rows)
        self.trial_hashes = sorted(item_ids_by_trial)
        self.active_evaluation_hash = None
        self._summaries = [
            _StubSummary(trial_hash, item_id)
            for trial_hash, item_id in item_ids_by_trial.items()
        ]

    def iter_trial_summaries(self):
        yield from self._summaries


def test_catalog_lists_humaneval_benchmarks() -> None:
    benchmark_ids = catalog.list_catalog_benchmarks()
    assert "humaneval" in benchmark_ids
    assert "humaneval_plus" in benchmark_ids


def test_humaneval_builder_uses_expected_defaults() -> None:
    definition = catalog.get_catalog_benchmark("humaneval")

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == "humaneval"
    assert benchmark.num_samples == 1
    assert benchmark.slices[0].scores[0].metrics == ["humaneval_pass_rate"]
    assert definition.metadata["variant"] == "base"
    assert definition.metadata["version"] == "v0.1.10"


def test_humaneval_builder_uses_requested_num_samples() -> None:
    definition = catalog.get_catalog_benchmark("humaneval_plus")

    benchmark = definition.build_benchmark(
        model_id="demo-model",
        provider="demo",
        num_samples=5,
    )

    assert benchmark.num_samples == 5


def test_humaneval_builtin_project_uses_loader_and_variant_settings(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def _loader(
        *,
        mini: bool,
        noextreme: bool,
        version: str,
        override_path: str | None = None,
        download: bool = True,
    ) -> list[dict[str, object]]:
        seen["mini"] = mini
        seen["noextreme"] = noextreme
        seen["version"] = version
        seen["override_path"] = override_path
        seen["download"] = download
        return [_humaneval_row()]

    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="humaneval_plus:mini,v0.1.10",
            model_id="demo-model",
            provider="demo",
            storage_root=tmp_path / "humaneval-default-loader",
            subset=1,
            huggingface_loader=_loader,
        )
    )
    planner = TrialPlanner(
        dataset_provider=dataset_provider,
        registry=registry,
        project_seed=project.global_seed,
    )

    planned_trials = planner.plan_benchmark(benchmark)

    assert len(planned_trials) == 1
    assert seen == {
        "mini": True,
        "noextreme": False,
        "version": "v0.1.10",
        "override_path": None,
        "download": True,
    }


def test_humaneval_provider_normalizes_base_and_plus_rows() -> None:
    base_definition = catalog.get_catalog_benchmark("humaneval")
    plus_definition = catalog.get_catalog_benchmark("humaneval_plus")
    base_benchmark = base_definition.build_benchmark(
        model_id="demo-model", provider="demo"
    )
    plus_benchmark = plus_definition.build_benchmark(
        model_id="demo-model", provider="demo"
    )
    provider = BuiltinHumanEvalDatasetProvider(huggingface_loader=lambda **kwargs: [])

    base_prepared = provider.prepare_rows([_humaneval_row()], base_benchmark.slices[0])
    plus_prepared = provider.prepare_rows([_humaneval_row()], plus_benchmark.slices[0])
    base_row = base_prepared.rows[0]
    plus_row = plus_prepared.rows[0]
    base_tests = cast(list[dict[str, str]], base_row["official_tests"])
    plus_tests = cast(list[dict[str, str]], base_row["plus_tests"])

    assert [row["item_id"] for row in base_prepared.rows] == ["HumanEval/0"]
    assert base_row["language"] == "python"
    assert base_row["execution_mode"] == "function"
    assert base_row["function_name"] == "add"
    assert len(base_tests) == 2
    assert len(plus_tests) == 1
    assert plus_row["score_variant"] == "plus"


def test_humaneval_metric_scores_plus_candidate_with_fenced_code() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout=json.dumps(
                    {
                        "base_status": "pass",
                        "plus_status": "pass",
                        "base_passed": 2,
                        "base_total": 2,
                        "plus_passed": 1,
                        "plus_total": 1,
                    }
                ),
                stderr="",
                return_code=0,
                status="ok",
            )
        ]
    )
    metric = HumanEvalExecutionMetric(
        metric_id="humaneval_plus_pass_rate", executor=executor
    )
    candidate = SimpleNamespace(
        inference=SimpleNamespace(
            raw_text=(
                "Here is the solution:\n```python\n"
                "def add(a, b):\n"
                "    return a + b\n"
                "```"
            )
        )
    )

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "execution_mode": "function",
            "function_name": "add",
            "score_variant": "plus",
            "official_tests": [
                {"input": "[1, 2]", "output": "3"},
                {"input": "[4, 5]", "output": "9"},
            ],
            "plus_tests": [{"input": "[10, 20]", "output": "30"}],
            "atol": 0.0,
            "base_expected": [3, 9],
            "plus_expected": [30],
            "base_time_limits": [0.1, 0.1],
            "plus_time_limits": [0.1],
        },
    )

    assert score.metric_id == "humaneval_plus_pass_rate"
    assert score.value == 1.0
    assert score.details["base_status"] == "pass"
    assert score.details["plus_status"] == "pass"
    assert executor.calls[0]["language"] == "python"
    assert executor.calls[0]["timeout_seconds"] == pytest.approx(1.3)
    assert "def add(a, b)" in executor.calls[0]["code"]


def test_humaneval_metric_handles_float_tolerance() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout=json.dumps(
                    {
                        "base_status": "pass",
                        "plus_status": None,
                        "base_passed": 1,
                        "base_total": 1,
                        "plus_passed": 0,
                        "plus_total": 0,
                    }
                ),
                stderr="",
                return_code=0,
                status="ok",
            )
        ]
    )
    metric = HumanEvalExecutionMetric(
        metric_id="humaneval_pass_rate", executor=executor
    )
    candidate = SimpleNamespace(
        inference=SimpleNamespace(raw_text="def add(a, b):\n    return a + b\n")
    )

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "execution_mode": "function",
            "function_name": "add",
            "score_variant": "base",
            "official_tests": [{"input": "[0.1, 0.2]", "output": "0.3"}],
            "plus_tests": [],
            "atol": 1e-6,
            "base_expected": [0.3],
            "plus_expected": [],
            "base_time_limits": [0.05],
            "plus_time_limits": [],
        },
    )

    assert score.value == 1.0
    assert score.details["base_passed"] == 1


def test_humaneval_metric_handles_find_zero_special_oracle() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout=json.dumps(
                    {
                        "base_status": "pass",
                        "plus_status": None,
                        "base_passed": 1,
                        "base_total": 1,
                        "plus_passed": 0,
                        "plus_total": 0,
                    }
                ),
                stderr="",
                return_code=0,
                status="ok",
            )
        ]
    )
    metric = HumanEvalExecutionMetric(
        metric_id="humaneval_pass_rate", executor=executor
    )
    candidate = SimpleNamespace(
        inference=SimpleNamespace(raw_text="def find_zero(xs):\n    return 1.0\n")
    )

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "execution_mode": "function",
            "function_name": "find_zero",
            "score_variant": "base",
            "official_tests": [{"input": "[[1.0, 0.0, -1.0]]", "output": "0.0"}],
            "plus_tests": [],
            "atol": 1e-6,
            "base_expected": [0.0],
            "plus_expected": [],
            "base_time_limits": [0.05],
            "plus_time_limits": [],
        },
    )

    assert score.value == 1.0
    assert "find_zero" in executor.calls[0]["code"]


def test_humaneval_metric_renders_python_literals_in_harness() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout=json.dumps(
                    {
                        "base_status": "pass",
                        "plus_status": None,
                        "base_passed": 2,
                        "base_total": 2,
                        "plus_passed": 0,
                        "plus_total": 0,
                    }
                ),
                stderr="",
                return_code=0,
                status="ok",
            )
        ]
    )
    metric = HumanEvalExecutionMetric(
        metric_id="humaneval_pass_rate", executor=executor
    )
    candidate = SimpleNamespace(
        inference=SimpleNamespace(raw_text="def choose(flag):\n    return flag\n")
    )

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "execution_mode": "function",
            "function_name": "choose",
            "score_variant": "base",
            "official_tests": [
                {"input": "[true]", "output": "true"},
                {"input": "[false]", "output": "false"},
            ],
            "plus_tests": [],
            "atol": 0.0,
            "base_expected": [True, False],
            "plus_expected": [],
            "base_time_limits": [0.05, 0.05],
            "plus_time_limits": [],
        },
    )

    assert score.value == 1.0
    assert "BASE_INPUTS = [[True], [False]]" in executor.calls[0]["code"]
    assert "BASE_EXPECTED = [True, False]" in executor.calls[0]["code"]
    assert "true" not in executor.calls[0]["code"]


def test_humaneval_summary_computes_pass_at_k() -> None:
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="humaneval_pass_rate",
            score=1.0,
            details={"base_status": "pass", "plus_status": None},
        ),
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-2",
            metric_id="humaneval_pass_rate",
            score=0.0,
            details={"base_status": "fail", "plus_status": None},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-1",
            metric_id="humaneval_pass_rate",
            score=0.0,
            details={"base_status": "fail", "plus_status": None},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="humaneval_pass_rate",
            score=1.0,
            details={"base_status": "pass", "plus_status": None},
        ),
    ]
    result = _StubResult(rows, {"trial-1": "HumanEval/0", "trial-2": "HumanEval/1"})

    summary = summarize_humaneval(None, result)
    base_pass_at_k = cast(dict[str, object], summary["base_pass_at_k"])

    assert summary["metric_id"] == "humaneval_pass_rate"
    assert summary["sample_count_min"] == 2
    assert base_pass_at_k["pass@1"] == pytest.approx(0.5)


def test_humaneval_plus_summary_requires_base_and_plus_pass() -> None:
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="humaneval_plus_pass_rate",
            score=1.0,
            details={"base_status": "pass", "plus_status": "pass"},
        ),
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-2",
            metric_id="humaneval_plus_pass_rate",
            score=0.0,
            details={"base_status": "pass", "plus_status": "fail"},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-1",
            metric_id="humaneval_plus_pass_rate",
            score=0.0,
            details={"base_status": "fail", "plus_status": "fail"},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="humaneval_plus_pass_rate",
            score=1.0,
            details={"base_status": "pass", "plus_status": "pass"},
        ),
    ]
    result = _StubResult(rows, {"trial-1": "HumanEval/0", "trial-2": "HumanEval/1"})

    summary = summarize_humaneval_plus(None, result)
    base_pass_at_k = cast(dict[str, object], summary["base_pass_at_k"])
    plus_pass_at_k = cast(dict[str, object], summary["plus_pass_at_k"])

    assert summary["metric_id"] == "humaneval_plus_pass_rate"
    assert base_pass_at_k["pass@1"] == pytest.approx(0.75)
    assert plus_pass_at_k["pass@1"] == pytest.approx(0.5)


def test_humaneval_download_helper_honors_override_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override_path = tmp_path / "HumanEvalPlus.jsonl"
    override_path.write_text(json.dumps(_humaneval_row()) + "\n")
    monkeypatch.setenv("HUMANEVAL_OVERRIDE_PATH", str(override_path))

    rows = humaneval_dataset.load_humaneval_plus_rows(
        mini=False,
        noextreme=False,
        version="v0.1.10",
    )
    first_row = rows[0]

    assert first_row["task_id"] == "HumanEval/0"


def test_humaneval_loader_handles_large_integer_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    digits = "9" * 5000
    override_path = tmp_path / "humaneval-large-int.jsonl"
    row = _humaneval_row(base_input=[], plus_input=[])
    payload = json.dumps(row)
    payload = payload.replace('"base_input": []', f'"base_input": [[{digits}]]', 1)
    override_path.write_text(payload + "\n")
    monkeypatch.setenv("HUMANEVAL_OVERRIDE_PATH", str(override_path))

    rows = humaneval_dataset.load_humaneval_plus_rows(
        mini=False,
        noextreme=False,
        version="v0.1.10",
        download=False,
    )
    first_row = rows[0]
    base_input = cast(list[list[object]], first_row["base_input"])
    value = base_input[0][0]

    assert isinstance(value, int)
    assert value > 0
    assert value % 10 == 9
    assert value.bit_length() > 16000


def test_humaneval_normalization_handles_large_integer_payloads() -> None:
    huge_integer = humaneval_dataset._json_loads_unbounded("9" * 5000)
    row = _humaneval_row(
        base_input=[[huge_integer]],
        plus_input=[],
        base_expected=[huge_integer],
        plus_expected=[],
        base_time_limits=[0.05],
        plus_time_limits=[],
    )

    normalized = humaneval_dataset._normalize_humaneval_row(row, score_variant="base")
    official_tests = cast(list[dict[str, str]], normalized["official_tests"])

    assert len(official_tests) == 1
    assert official_tests[0]["input"].startswith("[")
    assert len(official_tests[0]["input"]) > 5000
