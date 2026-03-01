"""Integration tests for CLI comparison commands."""

from __future__ import annotations

import json
from types import SimpleNamespace


from themis.cli.commands.comparison_commands import (
    compare_command,
    _generate_comparison_html,
    _generate_comparison_markdown,
)


# ---------------------------------------------------------------------------
# compare_command
# ---------------------------------------------------------------------------


class TestCompareCommand:
    """Tests for the `themis compare` CLI command."""

    def test_compare_too_few_runs(self, capsys):
        exit_code = compare_command(["only-one"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "at least 2 runs" in captured.err

    def test_compare_storage_not_found(self, tmp_path, capsys):
        nonexistent = tmp_path / "nope"
        exit_code = compare_command(["run-a", "run-b"], storage=str(nonexistent))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_compare_success(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        class FakeReport:
            def summary(self, *, include_details=False):
                return "FAKE SUMMARY"

            def to_dict(self):
                return {"ok": True}

        def fake_compare_runs(**kwargs):
            assert set(kwargs["run_ids"]) == {"run-a", "run-b"}
            return FakeReport()

        monkeypatch.setattr(
            "themis.experiment.comparison.compare_runs", fake_compare_runs
        )

        exit_code = compare_command(["run-a", "run-b"], storage=str(storage_dir))
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Comparing 2 runs" in captured.out
        assert "FAKE SUMMARY" in captured.out

    def test_compare_with_json_output(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()
        output_file = tmp_path / "report.json"

        class FakeReport:
            def summary(self, *, include_details=False):
                return "ok"

            def to_dict(self):
                return {"metric": "exact_match", "best": "run-a"}

        def fake_compare_runs(**kwargs):
            return FakeReport()

        monkeypatch.setattr(
            "themis.experiment.comparison.compare_runs", fake_compare_runs
        )

        exit_code = compare_command(
            ["run-a", "run-b"],
            storage=str(storage_dir),
            output=str(output_file),
        )
        assert exit_code == 0
        data = json.loads(output_file.read_text())
        assert data["metric"] == "exact_match"

    def test_compare_with_html_output(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()
        output_file = tmp_path / "report.html"

        class FakeReport:
            run_ids = ["run-a", "run-b"]
            metrics = ["exact_match"]
            overall_best_run = "run-a"
            best_run_per_metric = {"exact_match": "run-a"}
            win_loss_matrices = {}

            def summary(self, *, include_details=False):
                return "ok"

            def to_dict(self):
                return {}

        def fake_compare_runs(**kwargs):
            return FakeReport()

        monkeypatch.setattr(
            "themis.experiment.comparison.compare_runs", fake_compare_runs
        )

        exit_code = compare_command(
            ["run-a", "run-b"],
            storage=str(storage_dir),
            output=str(output_file),
        )
        assert exit_code == 0
        html = output_file.read_text()
        assert "<html>" in html
        assert "Comparison Report" in html

    def test_compare_with_markdown_output(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()
        output_file = tmp_path / "report.md"

        class FakeReport:
            run_ids = ["run-a", "run-b"]
            metrics = ["exact_match"]
            overall_best_run = "run-a"
            best_run_per_metric = {"exact_match": "run-a"}
            win_loss_matrices = {}

            def summary(self, *, include_details=False):
                return "ok"

            def to_dict(self):
                return {}

        def fake_compare_runs(**kwargs):
            return FakeReport()

        monkeypatch.setattr(
            "themis.experiment.comparison.compare_runs", fake_compare_runs
        )

        exit_code = compare_command(
            ["run-a", "run-b"],
            storage=str(storage_dir),
            output=str(output_file),
        )
        assert exit_code == 0
        md = output_file.read_text()
        assert "# Comparison Report" in md

    def test_compare_with_show_diff(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        class FakeReport:
            def summary(self, *, include_details=False):
                assert include_details is True
                return "diff details"

            def to_dict(self):
                return {}

        def fake_compare_runs(**kwargs):
            return FakeReport()

        monkeypatch.setattr(
            "themis.experiment.comparison.compare_runs", fake_compare_runs
        )

        exit_code = compare_command(
            ["run-a", "run-b"],
            storage=str(storage_dir),
            show_diff=True,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "diff details" in captured.out

    def test_compare_engine_error(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        def fake_compare_runs(**kwargs):
            raise RuntimeError("storage corrupted")

        monkeypatch.setattr(
            "themis.experiment.comparison.compare_runs", fake_compare_runs
        )

        exit_code = compare_command(["run-a", "run-b"], storage=str(storage_dir))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "storage corrupted" in captured.err


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


class TestReportGenerators:
    """Tests for the internal HTML/Markdown report generators."""

    def _make_report(self):
        matrix = SimpleNamespace(
            run_ids=["run-a", "run-b"],
            matrix=[["—", "win"], ["loss", "—"]],
        )
        return SimpleNamespace(
            run_ids=["run-a", "run-b"],
            metrics=["exact_match"],
            overall_best_run="run-a",
            best_run_per_metric={"exact_match": "run-a"},
            win_loss_matrices={"exact_match": matrix},
        )

    def test_generate_html(self):
        report = self._make_report()
        html = _generate_comparison_html(report)
        assert "<html>" in html
        assert "run-a" in html
        assert "run-b" in html
        assert "exact_match" in html
        assert "Comparison Report" in html

    def test_generate_markdown(self):
        report = self._make_report()
        md = _generate_comparison_markdown(report)
        assert "# Comparison Report" in md
        assert "run-a" in md
        assert "run-b" in md
        assert "exact_match" in md
