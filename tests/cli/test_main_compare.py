from __future__ import annotations

from pathlib import Path

from themis.cli import main as cli_main


def test_compare_help_runs_without_type_hint_crash():
    exit_code = cli_main.main(["compare", "--help"])
    assert exit_code == 0


def test_compare_command_invokes_engine(tmp_path, monkeypatch, capsys):
    class _FakeReport:
        def summary(self, *, include_details: bool = False) -> str:
            assert include_details is False
            return "ok"

        def to_dict(self) -> dict[str, object]:
            return {"ok": True}

    def _fake_compare_runs(**kwargs):
        assert kwargs["run_ids"] == ["run-a", "run-b"]
        assert Path(kwargs["storage_path"]) == tmp_path
        return _FakeReport()

    monkeypatch.setattr("themis.comparison.compare_runs", _fake_compare_runs)

    exit_code = cli_main.compare(
        ["run-a", "run-b"],
        storage=str(tmp_path),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Comparing 2 runs: run-a, run-b" in captured.out
    assert "ok" in captured.out
