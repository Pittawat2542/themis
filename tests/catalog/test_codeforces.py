from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import themis.catalog as catalog
from themis import Orchestrator
from themis.catalog.benchmarks.codeforces.dataset import (
    BuiltinOpenR1CodeforcesDatasetProvider,
)
from themis.catalog.benchmarks.codeforces import dataset as codeforces_dataset
from themis.catalog.benchmarks.codeforces.metric import (
    CodeforcesExecutionMetric,
    PistonSandboxExecutor,
    SandboxFusionExecutor,
    SandboxExecutionResult,
    _default_executor,
    _resolve_piston_runtime,
)
from themis.contracts.protocols import InferenceResult
from themis.orchestration.trial_planner import TrialPlanner
from themis.records import InferenceRecord


def _sample_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "123A",
        "contest_id": "123",
        "title": "Fixture Problem",
        "description": "Solve it.",
        "input_format": "Read input.",
        "output_format": "Write output.",
        "interaction_format": None,
        "time_limit": 2.0,
        "memory_limit": 256.0,
        "official_tests_complete": True,
        "official_tests": [
            {"input": "1\n", "output": "2\n"},
            {"input": "4\n", "output": "5\n"},
        ],
        "input_mode": "stdio",
        "generated_checker": None,
        "executable": True,
        "generated_tests": 0,
        "language": "python",
        "prompt": "Write a Python program.",
        "rating": 1200,
        "tags": ["implementation"],
    }
    row.update(overrides)
    return row


def test_catalog_lists_codeforces() -> None:
    assert "codeforces" in catalog.list_catalog_benchmarks()


def test_codeforces_builder_uses_expected_defaults() -> None:
    definition = catalog.get_catalog_benchmark("codeforces")

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == "codeforces"
    assert benchmark.slices[0].dataset.dataset_id == "open-r1/codeforces"
    assert benchmark.slices[0].dataset.split == "test"
    assert benchmark.slices[0].scores[0].metrics == ["codeforces_pass_rate"]
    rename_transform = benchmark.slices[0].dataset.transforms[0]
    assert rename_transform.field == "prompt_text"
    assert getattr(rename_transform, "source_field", None) == "prompt"


def test_codeforces_builtin_project_uses_verifiable_prompts_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class _FakeDataset:
        features: dict[str, object] = {}

        def __iter__(self):
            yield _sample_row()

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
        codeforces_dataset,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="codeforces",
            model_id="demo-model",
            provider="demo",
            storage_root=tmp_path / "codeforces-default-loader",
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
        "dataset_id": "open-r1/codeforces",
        "config_name": "verifiable-prompts",
        "split": "test",
        "revision": None,
    }


def test_codeforces_provider_filters_unsupported_rows() -> None:
    definition = catalog.get_catalog_benchmark("codeforces")
    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")
    provider = BuiltinOpenR1CodeforcesDatasetProvider(
        huggingface_loader=lambda dataset_id, split, revision: []
    )

    prepared = provider.prepare_rows(
        [
            _sample_row(id="supported"),
            _sample_row(id="file-mode", input_mode="file"),
            _sample_row(id="interactive", interaction_format="interactive"),
            _sample_row(id="not-complete", official_tests_complete=False),
        ],
        benchmark.slices[0],
    )

    assert [row["item_id"] for row in prepared.rows] == ["supported"]
    assert prepared.rows[0]["prompt_text"] == "Write a Python program."
    assert prepared.rows[0]["metadata"] == {
        "contest_id": "123",
        "language": "python",
        "rating": "1200",
        "input_mode": "stdio",
    }
    assert prepared.stats == {
        "skipped_file_mode_count": 1,
        "skipped_interactive_count": 1,
        "skipped_incomplete_tests_count": 1,
    }


def test_codeforces_provider_rejects_rows_missing_required_fields() -> None:
    definition = catalog.get_catalog_benchmark("codeforces")
    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")
    provider = BuiltinOpenR1CodeforcesDatasetProvider(
        huggingface_loader=lambda dataset_id, split, revision: []
    )

    with pytest.raises(ValueError, match="codeforces.*prompt"):
        provider.prepare_rows(
            [_sample_row(prompt=None)],
            benchmark.slices[0],
        )

    with pytest.raises(ValueError, match="codeforces.*language"):
        provider.prepare_rows(
            [_sample_row(language="")],
            benchmark.slices[0],
        )


def test_codeforces_planner_preserves_prompt_and_language_payload() -> None:
    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="codeforces",
            model_id="demo-model",
            provider="demo",
            storage_root=Path(".cache/themis-tests/codeforces-planner"),
            subset=1,
            huggingface_loader=lambda dataset_id, split, revision: [_sample_row()],
        )
    )

    planner = TrialPlanner(
        dataset_provider=dataset_provider,
        registry=registry,
        project_seed=project.global_seed,
    )
    planned_trials = planner.plan_benchmark(benchmark)

    assert len(planned_trials) == 1
    payload = planned_trials[0].dataset_context.payload
    assert payload["prompt"] == "Write a Python program."
    assert payload["prompt_text"] == "Write a Python program."
    assert payload["language"] == "python"


def test_codeforces_orchestrator_uses_prompt_and_language_from_row(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class CapturingEngine:
        def infer(self, trial, context, runtime):
            del runtime
            seen["prompt"] = trial.prompt.messages[0].content
            seen["language"] = context["language"]
            return InferenceResult(
                inference=InferenceRecord(
                    spec_hash=f"inf_{context['item_id']}",
                    raw_text="candidate-output",
                )
            )

    class PassingExecutor:
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
            del code, stdin, args, timeout_seconds, memory_limit_mb
            if files:
                assert language == "python"
                return SandboxExecutionResult(
                    stdout="1\n",
                    stderr="",
                    return_code=0,
                    status="ok",
                )
            assert language == "python"
            return SandboxExecutionResult(
                stdout="candidate-output\n",
                stderr="",
                return_code=0,
                status="ok",
            )

    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="codeforces",
            model_id="demo-model",
            provider="demo",
            storage_root=tmp_path / "codeforces-offline",
            subset=1,
            huggingface_loader=lambda dataset_id, split, revision: [
                _sample_row(
                    generated_checker=(
                        "import sys\n"
                        "solution = open(sys.argv[3]).read().strip()\n"
                        'print(1 if solution == "candidate-output" else 0)'
                    )
                )
            ],
        )
    )
    registry.register_inference_engine("demo", CapturingEngine())
    registry.register_metric(
        "codeforces_pass_rate",
        lambda: CodeforcesExecutionMetric(executor=PassingExecutor()),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=dataset_provider,
    )

    result = orchestrator.run_benchmark(benchmark)

    assert seen == {
        "prompt": "Write a Python program.",
        "language": "python",
    }
    assert result.aggregate(group_by=["model_id", "slice_id", "metric_id"]) == [
        {
            "count": 1,
            "mean": 1.0,
            "metric_id": "codeforces_pass_rate",
            "model_id": "demo-model",
            "slice_id": "codeforces",
        }
    ]


def test_codeforces_orchestrator_fails_fast_on_missing_language(
    tmp_path: Path,
) -> None:
    project, benchmark, registry, dataset_provider, _definition = (
        catalog.build_catalog_benchmark_project(
            benchmark_id="codeforces",
            model_id="demo-model",
            provider="demo",
            storage_root=tmp_path / "codeforces-bad-row",
            subset=1,
            huggingface_loader=lambda dataset_id, split, revision: [
                _sample_row(language=None)
            ],
        )
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=dataset_provider,
    )

    with pytest.raises(ValueError, match="codeforces.*language"):
        orchestrator.run_benchmark(benchmark)


class _FakeExecutor:
    def __init__(self, results: list[SandboxExecutionResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

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


def test_codeforces_metric_scores_passing_stdio_solution() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout="2\n",
                stderr="",
                return_code=0,
                status="ok",
            ),
            SandboxExecutionResult(
                stdout="5\n",
                stderr="",
                return_code=0,
                status="ok",
            ),
        ]
    )
    metric = CodeforcesExecutionMetric(executor=executor)
    candidate = SimpleNamespace(inference=SimpleNamespace(raw_text="print(1)"))

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "input_mode": "stdio",
            "time_limit": 2.0,
            "memory_limit": 256.0,
            "official_tests": [
                {"input": "1\n", "output": "2\n"},
                {"input": "4\n", "output": "5\n"},
            ],
        },
    )

    assert score.metric_id == "codeforces_pass_rate"
    assert score.value == 1.0
    assert score.details["passed_tests"] == 2
    assert score.details["total_tests"] == 2
    assert score.details["used_checker"] is False
    assert executor.calls[0]["timeout_seconds"] == pytest.approx(2.0)
    assert executor.calls[0]["memory_limit_mb"] == pytest.approx(256.0)


def test_codeforces_metric_uses_generated_checker_when_present() -> None:
    executor = _FakeExecutor(
        [
            SandboxExecutionResult(
                stdout="candidate-output\n",
                stderr="",
                return_code=0,
                status="ok",
            ),
            SandboxExecutionResult(
                stdout="1\n",
                stderr="",
                return_code=0,
                status="ok",
            ),
        ]
    )
    metric = CodeforcesExecutionMetric(executor=executor)
    candidate = SimpleNamespace(inference=SimpleNamespace(raw_text="print(1)"))

    score = metric.score(
        None,
        candidate,
        {
            "language": "python",
            "input_mode": "stdio",
            "time_limit": 1.5,
            "memory_limit": 128.0,
            "official_tests": [{"input": "1\n", "output": "2\n"}],
            "generated_checker": "print(1)",
        },
    )

    assert score.value == 1.0
    assert score.details["passed_tests"] == 1
    assert score.details["used_checker"] is True
    assert len(executor.calls) == 2
    assert executor.calls[1]["language"] == "python"
    assert executor.calls[1]["args"] == [
        "input.txt",
        "correct_output.txt",
        "solution_output.txt",
    ]
    assert executor.calls[1]["files"] == {
        "checker.py": "print(1)",
        "input.txt": "1\n",
        "correct_output.txt": "2\n",
        "solution_output.txt": "candidate-output\n",
    }


def test_sandbox_fusion_executor_sends_stdin_and_base64_files() -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"status":"Success","message":"","compile_result":null,'
                b'"run_result":{"status":"Finished","return_code":0,'
                b'"stdout":"ok\\n","stderr":""}}'
            )

    def _fake_urlopen(req):
        captured["url"] = req.full_url
        captured["body"] = req.data.decode("utf-8")
        return _Response()

    executor = SandboxFusionExecutor(
        base_url="http://localhost:8080",
        urlopen=_fake_urlopen,
    )

    result = executor.execute(
        code="print(1)",
        language="python",
        stdin="abc\n",
        files={"input.txt": "hello\n"},
    )

    assert result.ok is True
    assert result.stdout == "ok\n"
    assert captured["url"] == "http://localhost:8080/run_code"
    assert '"stdin": "abc\\n"' in str(captured["body"])
    assert '"input.txt": "aGVsbG8K"' in str(captured["body"])


def test_sandbox_fusion_executor_wraps_python_args() -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"status":"Success","message":"","compile_result":null,'
                b'"run_result":{"status":"Finished","return_code":0,'
                b'"stdout":"ok\\n","stderr":""}}'
            )

    def _fake_urlopen(req):
        captured["body"] = req.data.decode("utf-8")
        return _Response()

    executor = SandboxFusionExecutor(
        base_url="http://localhost:8080",
        urlopen=_fake_urlopen,
    )

    executor.execute(
        code="print('unused')",
        language="python",
        files={"checker.py": "print('checker')"},
        args=["input.txt", "correct_output.txt", "solution_output.txt"],
    )

    body = str(captured["body"])
    assert "sys.argv =" in body
    assert "checker.py" in body
    assert "solution_output.txt" in body
    assert '"checker.py": "cHJpbnQoJ2NoZWNrZXInKQ=="' in body


def test_piston_executor_prefers_generic_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THEMIS_CODE_PISTON_URL", "http://generic-piston:2000")

    executor = PistonSandboxExecutor()

    assert executor._base_url == "http://generic-piston:2000"


def test_sandbox_fusion_executor_prefers_generic_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THEMIS_CODE_SANDBOX_FUSION_URL", "http://generic-sf:8080")

    executor = SandboxFusionExecutor()

    assert executor._base_url == "http://generic-sf:8080"


def test_default_executor_uses_generic_sandbox_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THEMIS_CODE_SANDBOX", "sandbox_fusion")

    executor = _default_executor()

    assert isinstance(executor, SandboxFusionExecutor)


def test_default_executor_ignores_legacy_sandbox_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THEMIS_CODE_SANDBOX", raising=False)
    monkeypatch.setenv("THEMIS_CODEFORCES_SANDBOX", "sandbox_fusion")

    executor = _default_executor()

    assert isinstance(executor, PistonSandboxExecutor)


def test_resolve_piston_runtime_ignores_legacy_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THEMIS_CODEFORCES_PISTON_PYTHON_LANGUAGE", "python")
    monkeypatch.setenv("THEMIS_CODEFORCES_PISTON_PYTHON_VERSION", "3.12.0")

    runtime = _resolve_piston_runtime("python")

    assert runtime == ("python", "*")
