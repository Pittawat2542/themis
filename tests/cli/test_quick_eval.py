from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest

from themis.errors import ThemisError
from themis.cli import quick_eval as quick_eval_cli
from themis.cli.main import main
from themis.types.enums import ErrorCode


def test_format_display_path_normalizes_windows_separators() -> None:
    assert (
        quick_eval_cli._format_display_path(
            PureWindowsPath(r".cache\themis\quick-eval\inline-demo-model-exact-match")
        )
        == ".cache/themis/quick-eval/inline-demo-model-exact-match"
    )


def test_quick_eval_inline_preview_renders_default_prompt(
    tmp_path: Path, capsys
) -> None:
    storage_root = tmp_path / "inline-preview"

    assert (
        main(
            [
                "quick-eval",
                "inline",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--input",
                "2 + 2",
                "--expected",
                "4",
                "--preview",
                "--format",
                "json",
                "--storage-root",
                str(storage_root),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "inline"
    assert payload["preview"][0]["messages"][0]["content"] == "2 + 2"
    assert payload["storage_root"] == storage_root.as_posix()


def test_quick_eval_file_run_emits_json_results(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "qa.csv"
    dataset_path.write_text("item_id,question,answer\nq1,2 + 2,4\nq2,3 + 3,6\n")
    storage_root = tmp_path / "file-run"

    assert (
        main(
            [
                "quick-eval",
                "file",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--file",
                str(dataset_path),
                "--input-field",
                "question",
                "--expected-field",
                "answer",
                "--item-id-field",
                "item_id",
                "--format",
                "json",
                "--storage-root",
                str(storage_root),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "file"
    assert payload["sqlite_db"] == (storage_root / "themis.sqlite3").as_posix()
    assert payload["rows"][0]["metric_id"] == "exact_match"
    assert payload["rows"][0]["mean"] == 1.0
    assert payload["rows"][0]["count"] == 2


def test_quick_eval_huggingface_estimate_uses_subset_count(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from themis.cli import quick_eval as quick_eval_cli

    monkeypatch.setattr(
        quick_eval_cli,
        "load_huggingface_rows",
        lambda dataset_id, split, revision: [
            {"item_id": "row-1", "question": "1 + 1", "answer": "2"},
            {"item_id": "row-2", "question": "2 + 2", "answer": "4"},
            {"item_id": "row-3", "question": "3 + 3", "answer": "6"},
        ],
    )

    assert (
        main(
            [
                "quick-eval",
                "huggingface",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--dataset-id",
                "demo/arithmetic",
                "--split",
                "test",
                "--input-field",
                "question",
                "--expected-field",
                "answer",
                "--subset",
                "2",
                "--estimate-only",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "hf-run"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "huggingface"
    assert payload["estimate"]["trial_count"] == 2


def test_quick_eval_huggingface_surfaces_missing_extra(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from themis.cli import quick_eval as quick_eval_cli

    def _missing_loader(dataset_id: str, split: str, revision: str | None):
        del dataset_id, split, revision
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=(
                "Optional dependency 'datasets' is required for this feature. "
                'Install it with `uv add "themis-eval[datasets]"`.'
            ),
        )

    monkeypatch.setattr(quick_eval_cli, "load_huggingface_rows", _missing_loader)

    assert (
        main(
            [
                "quick-eval",
                "huggingface",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--dataset-id",
                "demo/arithmetic",
                "--split",
                "test",
                "--input-field",
                "question",
                "--expected-field",
                "answer",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "hf-missing"),
            ]
        )
        == 1
    )

    assert "themis-eval[datasets]" in capsys.readouterr().err


def test_quick_eval_benchmark_preview_uses_catalog(capsys) -> None:
    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "mmlu_pro",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--preview",
                "--format",
                "json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "benchmark"
    assert payload["benchmark"] == "mmlu_pro"
    assert (
        "Return the best option letter only."
        in payload["preview"][0]["messages"][0]["content"]
    )


def test_quick_eval_benchmark_defaults_to_8192_max_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    from themis.cli import quick_eval as quick_eval_cli

    observed: dict[str, object] = {}

    class _StubDefinition:
        benchmark_id = "mmlu_pro"
        primary_metric_id = "choice_accuracy"

        def render_preview(self, **kwargs):
            del kwargs
            return [{"messages": [{"role": "user", "content": "preview"}]}]

    def _stub_build_catalog_benchmark_project(**kwargs):
        observed.update(kwargs)
        return object(), object(), object(), object(), _StubDefinition()

    monkeypatch.setattr(
        quick_eval_cli,
        "build_catalog_benchmark_project",
        _stub_build_catalog_benchmark_project,
    )

    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "mmlu_pro",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--preview",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "benchmark-preview"),
            ]
        )
        == 0
    )

    assert observed["max_tokens"] == 8192


def test_quick_eval_benchmark_forwards_num_samples(tmp_path: Path, monkeypatch) -> None:
    from themis.cli import quick_eval as quick_eval_cli

    observed: dict[str, object] = {}

    class _StubDefinition:
        benchmark_id = "humaneval"
        primary_metric_id = "humaneval_pass_rate"

        def render_preview(self, **kwargs):
            del kwargs
            return [{"messages": [{"role": "user", "content": "preview"}]}]

    def _stub_build_catalog_benchmark_project(**kwargs):
        observed.update(kwargs)
        return object(), object(), object(), object(), _StubDefinition()

    monkeypatch.setattr(
        quick_eval_cli,
        "build_catalog_benchmark_project",
        _stub_build_catalog_benchmark_project,
    )

    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "humaneval",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--num-samples",
                "5",
                "--preview",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "benchmark-preview"),
            ]
        )
        == 0
    )

    assert observed["num_samples"] == 5


def test_quick_eval_benchmark_estimate_uses_builtin_dataset_loader(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import themis.catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "load_huggingface_rows",
        lambda dataset_id, split, revision: [
            {"question_id": 1, "question": "2 + 2", "options": ["4"], "answer": "A"},
            {"question_id": 2, "question": "3 + 3", "options": ["6"], "answer": "A"},
            {"question_id": 3, "question": "4 + 4", "options": ["8"], "answer": "A"},
        ],
    )

    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "mmlu_pro",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--subset",
                "2",
                "--estimate-only",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "benchmark-estimate"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "mmlu_pro"
    assert payload["estimate"]["trial_count"] == 2


def test_quick_eval_benchmark_requires_explicit_judge_config(
    tmp_path: Path, capsys
) -> None:
    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "simpleqa_verified",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "missing-judge"),
            ]
        )
        == 1
    )

    assert "judge" in capsys.readouterr().err.lower()


def test_quick_eval_openai_compatible_uses_env_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from themis.cli import quick_eval as quick_eval_cli

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1")

    extras = quick_eval_cli._provider_model_extras("openai_compatible")

    assert extras["base_url"] == "http://127.0.0.1:1234/v1"


def test_quick_eval_math_benchmark_preview_uses_boxed_answer_prompt(capsys) -> None:
    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "aime_2026",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--preview",
                "--format",
                "json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "benchmark"
    assert payload["benchmark"] == "aime_2026"
    assert "\\boxed{" in payload["preview"][0]["messages"][0]["content"]


def test_quick_eval_math_benchmark_estimate_uses_builtin_dataset_loader(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import themis.catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "load_huggingface_rows",
        lambda dataset_id, split, revision: [
            {"problem_idx": 1, "problem": "2 + 2", "answer": "4"},
            {"problem_idx": 2, "problem": "3 + 3", "answer": "6"},
            {"problem_idx": 3, "problem": "4 + 4", "answer": "8"},
        ],
    )

    assert (
        main(
            [
                "quick-eval",
                "benchmark",
                "--benchmark",
                "aime_2026",
                "--model",
                "demo-model",
                "--provider",
                "demo",
                "--subset",
                "2",
                "--estimate-only",
                "--format",
                "json",
                "--storage-root",
                str(tmp_path / "benchmark-estimate-math"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "aime_2026"
    assert payload["estimate"]["trial_count"] == 2


def test_build_benchmark_summary_output_detects_humaneval_plus() -> None:
    summary_output = quick_eval_cli._build_benchmark_summary_output(
        "humaneval_plus:mini,v0.1.10",
        {
            "metric_id": "humaneval_plus_pass_rate",
            "task_count": 1,
            "sample_count_min": 1,
            "base_pass_at_k": {"pass@1": 1.0},
            "plus_pass_at_k": {"pass@1": 0.5},
        },
    )

    assert isinstance(summary_output, quick_eval_cli.HumanEvalSummaryOutput)
    assert summary_output.plus_pass_at_k == {"pass@1": 0.5}


def test_emit_quick_eval_output_renders_humaneval_plus_summary_table(
    capsys,
) -> None:
    quick_eval_cli._emit_quick_eval_output(
        {
            "mode": "benchmark",
            "benchmark": "humaneval_plus:mini,v0.1.10",
            "model": "demo-model",
            "provider": "demo",
            "metric": "humaneval_plus_pass_rate",
            "storage_root": "/tmp/demo",
            "rows": [
                {
                    "model_id": "demo-model",
                    "slice_id": "humaneval_plus:mini,v0.1.10",
                    "metric_id": "humaneval_plus_pass_rate",
                    "prompt_variant_id": "humaneval_plus:mini,v0.1.10-default",
                    "mean": 1.0,
                    "count": 1,
                }
            ],
            "summary": {
                "metric_id": "humaneval_plus_pass_rate",
                "task_count": 1,
                "sample_count_min": 1,
                "base_pass_at_k": {"pass@1": 1.0},
                "plus_pass_at_k": {"pass@1": 0.5},
            },
        },
        format="table",
    )

    output = capsys.readouterr().out

    assert "Pass@K" in output
    assert "base" in output
    assert "plus" in output
    assert "{'pass@1': 1.0}" not in output
