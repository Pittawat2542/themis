from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import zlib

import pytest

import themis.catalog as catalog
from themis.catalog.benchmarks.aethercode.dataset import (
    BuiltinAetherCodeDatasetProvider,
)
from themis.catalog.benchmarks.aethercode import dataset as aethercode_dataset
from themis.catalog.benchmarks.aethercode.metric import AetherCodeExecutionMetric
from themis.catalog.benchmarks.codeforces.metric import SandboxExecutionResult
from themis.catalog.benchmarks.livecodebench.dataset import (
    BuiltinLiveCodeBenchDatasetProvider,
    _decode_livecodebench_tests,
)
from themis.catalog.benchmarks.livecodebench.metric import LiveCodeBenchExecutionMetric
from themis.orchestration.trial_planner import TrialPlanner


def _encode_livecodebench_private_tests(tests: list[dict[str, str]]) -> str:
    payload = json.dumps(tests).encode("utf-8")
    compressed = zlib.compress(payload)
    return base64.b64encode(compressed).decode("ascii")


def _aethercode_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": 60173,
        "description": (
            "Input file: standard input\n"
            "Output file: standard output\n"
            "Time limit: 2 seconds\n"
            "Memory limit: 1024 megabytes\n\n"
            "Compute the answer."
        ),
        "time_limit": 2000,
        "memory_limit": 1024,
        "checker": (
            '#include "testlib.h"\n'
            "int main(int argc, char* argv[]) {\n"
            "  registerTestlibCmd(argc, argv);\n"
            '  quitf(_ok, "ok");\n'
            "}\n"
        ),
        "test_cases": [
            {"input": "1\n", "output": "2\n"},
            {"input": "4\n", "output": "5\n"},
        ],
        "year": 2024,
        "date": "2024/11/17",
        "difficulty": "Easy",
        "contest_category": "ICPC East Asia Regionals",
        "contest_name": "The 2024 ICPC Asia Shanghai Regional Contest",
    }
    row.update(overrides)
    return row


def _livecodebench_stdio_row(**overrides: object) -> dict[str, object]:
    private_tests = [
        {"input": "26", "output": "2025\n", "testtype": "stdin"},
        {"input": "1", "output": "2024\n", "testtype": "stdin"},
    ]
    row: dict[str, object] = {
        "question_title": "9x9 Sum",
        "question_content": "Read X and print the requested sum.",
        "platform": "atcoder",
        "question_id": "abc387_b",
        "contest_id": "abc387",
        "contest_date": "2025-01-04T00:00:00",
        "starter_code": "",
        "difficulty": "easy",
        "public_test_cases": json.dumps(
            [
                {"input": "1", "output": "2024", "testtype": "stdin"},
                {"input": "11", "output": "2025", "testtype": "stdin"},
            ]
        ),
        "private_test_cases": _encode_livecodebench_private_tests(private_tests),
        "metadata": "{}",
    }
    row.update(overrides)
    return row


def _livecodebench_functional_row(**overrides: object) -> dict[str, object]:
    private_tests = [
        {
            "input": "[[3,2],[3,2]]",
            "output": "[3,2]",
            "testtype": "functional",
        },
        {
            "input": "5\n[[1, 0, 1], [2, 0, 2]]\n2",
            "output": "1",
            "testtype": "functional",
        },
    ]
    row: dict[str, object] = {
        "question_title": "Fixture Function",
        "question_content": "Implement the requested method.",
        "platform": "leetcode",
        "question_id": "3708",
        "contest_id": "weekly-1",
        "contest_date": "2025-01-11T00:00:00",
        "starter_code": (
            "class Solution:\n"
            "    def zigzagTraversal(self, grid: list[list[int]]) -> list[int]:\n"
            "        "
        ),
        "difficulty": "medium",
        "public_test_cases": json.dumps(
            [
                {
                    "input": "[[1,2],[3,4]]",
                    "output": "[1, 4]",
                    "testtype": "functional",
                }
            ]
        ),
        "private_test_cases": _encode_livecodebench_private_tests(private_tests),
        "metadata": json.dumps({"func_name": "zigzagTraversal"}),
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


def test_livecodebench_decoder_accepts_compressed_json_payload() -> None:
    tests = [
        {"input": "1", "output": "2", "testtype": "stdin"},
        {"input": "3", "output": "4", "testtype": "functional"},
    ]

    decoded = _decode_livecodebench_tests(_encode_livecodebench_private_tests(tests))

    assert decoded == [
        {"input": "1", "output": "2"},
        {"input": "3", "output": "4", "testtype": "functional"},
    ]


def test_livecodebench_decoder_rejects_unknown_binary_payloads() -> None:
    legacy_payload = base64.b64encode(zlib.compress(b"\x80\x04legacy")).decode("ascii")

    with pytest.raises(ValueError, match="LiveCodeBench"):
        _decode_livecodebench_tests(legacy_payload)


def test_catalog_lists_new_code_benchmarks() -> None:
    benchmark_ids = catalog.list_catalog_benchmarks()
    assert "aethercode" in benchmark_ids
    assert "livecodebench" in benchmark_ids


def test_aethercode_builder_uses_expected_defaults() -> None:
    definition = catalog.get_catalog_benchmark("aethercode")

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == "aethercode"
    assert benchmark.slices[0].dataset.dataset_id == "m-a-p/AetherCode"
    assert benchmark.slices[0].dataset.split == "test"
    assert benchmark.slices[0].scores[0].metrics == ["aethercode_pass_rate"]
    assert definition.metadata["subset"] == "v1_2024"


def test_aethercode_builtin_project_uses_expected_subset_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class _FakeDataset:
        features: dict[str, object] = {}

        def __iter__(self):
            yield _aethercode_row()

    class _FakeDatasetsModule:
        @staticmethod
        def load_dataset(
            dataset_id: str,
            config_name: str,
            *,
            split: str,
            revision: str | None = None,
        ):
            seen["dataset_id"] = dataset_id
            seen["config_name"] = config_name
            seen["split"] = split
            seen["revision"] = revision
            return _FakeDataset()

    monkeypatch.setattr(
        aethercode_dataset,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="aethercode",
            model_id="demo-model",
            provider="demo",
            storage_root=tmp_path / "aethercode-default-loader",
            subset=1,
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
        "dataset_id": "m-a-p/AetherCode",
        "config_name": "v1_2024",
        "split": "test",
        "revision": None,
    }


def test_aethercode_loader_does_not_retry_streaming_for_unrelated_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDatasetsModule:
        class DatasetGenerationError(Exception):
            pass

        calls: list[dict[str, object]] = []

        @staticmethod
        def load_dataset(
            dataset_id: str,
            config_name: str,
            *,
            split: str,
            revision: str | None = None,
            streaming: bool = False,
        ):
            _FakeDatasetsModule.calls.append(
                {
                    "dataset_id": dataset_id,
                    "config_name": config_name,
                    "split": split,
                    "revision": revision,
                    "streaming": streaming,
                }
            )
            raise ValueError("bad config")

    monkeypatch.setattr(
        aethercode_dataset,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    with pytest.raises(ValueError, match="bad config"):
        aethercode_dataset._load_aethercode_rows("m-a-p/AetherCode", "test")

    assert _FakeDatasetsModule.calls == [
        {
            "dataset_id": "m-a-p/AetherCode",
            "config_name": "v1_2024",
            "split": "test",
            "revision": None,
            "streaming": False,
        }
    ]


def test_aethercode_provider_normalizes_rows() -> None:
    definition = catalog.get_catalog_benchmark("aethercode")
    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")
    provider = BuiltinAetherCodeDatasetProvider(
        huggingface_loader=lambda dataset_id, split, revision: []
    )

    prepared = provider.prepare_rows([_aethercode_row()], benchmark.slices[0])

    assert [row["item_id"] for row in prepared.rows] == ["60173"]
    assert prepared.rows[0]["language"] == "cpp"
    assert prepared.rows[0]["execution_mode"] == "stdio"
    assert prepared.rows[0]["official_tests"] == [
        {"input": "1\n", "output": "2\n"},
        {"input": "4\n", "output": "5\n"},
    ]
    assert prepared.rows[0]["generated_checker"]
    assert prepared.rows[0]["checker_language"] == "cpp"
    assert prepared.rows[0]["metadata"] == {
        "difficulty": "Easy",
        "contest_category": "ICPC East Asia Regionals",
        "contest_name": "The 2024 ICPC Asia Shanghai Regional Contest",
        "date": "2024/11/17",
        "year": "2024",
    }


def test_aethercode_metric_uses_cpp_checker_and_support_files() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout="2\n",
                stderr="",
                return_code=0,
                status="ok",
            ),
            SandboxExecutionResult(
                stdout="answer is 2\n",
                stderr="",
                return_code=0,
                status="ok",
            ),
        ]
    )
    metric = AetherCodeExecutionMetric(executor=executor)
    candidate = SimpleNamespace(inference=SimpleNamespace(raw_text="int main() {}"))

    score = metric.score(
        None,
        candidate,
        {
            "language": "cpp",
            "execution_mode": "stdio",
            "time_limit": 2.0,
            "memory_limit": 1024.0,
            "official_tests": [{"input": "1\n", "output": "2\n"}],
            "generated_checker": '#include "testlib.h"\nint main() { return 0; }\n',
            "checker_language": "cpp",
            "checker_support_files": {"testlib.h": "// fixture testlib header"},
        },
    )

    assert score.metric_id == "aethercode_pass_rate"
    assert score.value == 1.0
    assert executor.calls[1]["language"] == "cpp"
    assert executor.calls[1]["args"] == [
        "input.txt",
        "correct_output.txt",
        "solution_output.txt",
    ]
    assert executor.calls[1]["files"] == {
        "checker.cpp": '#include "testlib.h"\nint main() { return 0; }\n',
        "testlib.h": "// fixture testlib header",
        "input.txt": "1\n",
        "correct_output.txt": "2\n",
        "solution_output.txt": "2\n",
    }


def test_livecodebench_builder_uses_expected_defaults() -> None:
    definition = catalog.get_catalog_benchmark("livecodebench")

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == "livecodebench"
    assert (
        benchmark.slices[0].dataset.dataset_id == "livecodebench/code_generation_lite"
    )
    assert benchmark.slices[0].dataset.split == "test"
    assert benchmark.slices[0].scores[0].metrics == ["livecodebench_pass_rate"]
    assert definition.metadata["version_tag"] == "release_v6"


def test_livecodebench_builtin_project_uses_default_raw_loader(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def _loader(
        dataset_id: str,
        split: str,
        revision: str | None,
        *,
        version_tag: str,
    ) -> list[dict[str, object]]:
        seen["dataset_id"] = dataset_id
        seen["split"] = split
        seen["revision"] = revision
        seen["version_tag"] = version_tag
        return [_livecodebench_stdio_row()]

    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="livecodebench",
            model_id="demo-model",
            provider="demo",
            storage_root=tmp_path / "livecodebench-default-loader",
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
        "dataset_id": "livecodebench/code_generation_lite",
        "split": "test",
        "revision": None,
        "version_tag": "release_v6",
    }


def test_livecodebench_provider_normalizes_stdio_and_function_rows() -> None:
    definition = catalog.get_catalog_benchmark("livecodebench")
    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")
    provider = BuiltinLiveCodeBenchDatasetProvider(
        huggingface_loader=lambda dataset_id, split, revision, *, version_tag: []
    )

    prepared = provider.prepare_rows(
        [_livecodebench_stdio_row(), _livecodebench_functional_row()],
        benchmark.slices[0],
    )

    assert [row["item_id"] for row in prepared.rows] == ["abc387_b", "3708"]
    assert prepared.rows[0]["language"] == "python"
    assert prepared.rows[0]["execution_mode"] == "stdio"
    assert prepared.rows[0]["official_tests"] == [
        {"input": "26", "output": "2025\n"},
        {"input": "1", "output": "2024\n"},
    ]
    assert prepared.rows[1]["execution_mode"] == "function"
    assert prepared.rows[1]["function_name"] == "zigzagTraversal"
    official_tests = cast(list[dict[str, str]], prepared.rows[1]["official_tests"])
    assert official_tests[0] == {
        "input": "[[3,2],[3,2]]",
        "output": "[3,2]",
        "testtype": "functional",
    }


def test_livecodebench_metric_scores_functional_python_solution() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout="[3, 2]\n",
                stderr="",
                return_code=0,
                status="ok",
            )
        ]
    )
    metric = LiveCodeBenchExecutionMetric(executor=executor)
    candidate = SimpleNamespace(
        inference=SimpleNamespace(
            raw_text=(
                "class Solution:\n"
                "    def zigzagTraversal(self, grid):\n"
                "        return [grid[0][0], grid[1][1]]\n"
            )
        )
    )

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "execution_mode": "function",
            "function_name": "zigzagTraversal",
            "official_tests": [
                {
                    "input": "[[3,2],[3,2]]",
                    "output": "[3,2]",
                    "testtype": "functional",
                }
            ],
        },
    )

    assert score.metric_id == "livecodebench_pass_rate"
    assert score.value == 1.0
    assert executor.calls[0]["language"] == "python"
    assert executor.calls[0]["stdin"] == "[[3,2],[3,2]]"
    assert "zigzagTraversal" in executor.calls[0]["code"]
