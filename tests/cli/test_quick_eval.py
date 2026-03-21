from __future__ import annotations

import json
from pathlib import Path

from themis.errors import ThemisError
from themis.cli.main import main
from themis.types.enums import ErrorCode


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
    assert payload["storage_root"] == str(storage_root)


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
    assert payload["sqlite_db"] == str(storage_root / "themis.sqlite3")
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
