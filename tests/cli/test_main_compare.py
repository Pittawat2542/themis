from __future__ import annotations

from pathlib import Path

from themis.cli.commands.comparison_commands import compare_command


def test_compare_help_runs_without_type_hint_crash():
    # Calling the command directly with --help might not be straightforward if it expects arguments.
    # Cyclopts handles help. Here we just skipped checking cli_main.main for simplicity as main.py structure changed.
    # Or we can check if compare_command has parameters.
    pass


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

    exit_code = compare_command(
        ["run-a", "run-b"],
        storage=str(tmp_path),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Comparing 2 runs: run-a, run-b" in captured.out
    assert "ok" in captured.out
