"""Integration tests for CLI data commands (list, clean, share)."""

from __future__ import annotations

from types import SimpleNamespace


from themis.cli.commands.data_commands import (
    clean_command,
    list_command,
    share_command,
)


# ---------------------------------------------------------------------------
# list_command
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for `themis list` sub-command."""

    def test_list_invalid_what(self, capsys):
        exit_code = list_command("invalid_thing")
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not valid" in captured.out

    def test_list_benchmarks(self, capsys):
        exit_code = list_command("benchmarks")
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Available benchmarks:" in captured.out
        assert "demo" in captured.out

    def test_list_benchmarks_verbose(self, capsys):
        exit_code = list_command("benchmarks", verbose=True)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Available benchmarks:" in captured.out

    def test_list_benchmarks_limit(self, capsys):
        exit_code = list_command("benchmarks", limit=2)
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = [
            line for line in captured.out.splitlines() if line.strip().startswith("-")
        ]
        assert len(lines) <= 2

    def test_list_metrics(self, capsys):
        exit_code = list_command("metrics")
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Available metrics:" in captured.out
        assert "exact_match" in captured.out
        assert "bleu" in captured.out

    def test_list_runs_no_storage(self, tmp_path, capsys):
        nonexistent = tmp_path / "nope"
        exit_code = list_command("runs", storage=str(nonexistent))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No storage found" in captured.out

    def test_list_runs_empty_storage(self, tmp_path, capsys, monkeypatch):
        # Create an empty storage directory so the path exists
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        # Mock ExperimentStorage.list_runs to return empty
        class FakeStorage:
            def __init__(self, root):
                pass

            def list_runs(self, limit=None):
                return []

        monkeypatch.setattr("themis.storage.ExperimentStorage", FakeStorage)

        exit_code = list_command("runs", storage=str(storage_dir))
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No runs found" in captured.out

    def test_list_runs_with_results(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        run_meta = SimpleNamespace(
            run_id="test-run-1",
            status=SimpleNamespace(value="completed"),
            total_samples=10,
            created_at="2026-01-01T00:00:00",
        )

        class FakeStorage:
            def __init__(self, root):
                pass

            def list_runs(self, limit=None):
                return [run_meta]

        monkeypatch.setattr("themis.storage.ExperimentStorage", FakeStorage)

        exit_code = list_command("runs", storage=str(storage_dir))
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "test-run-1" in captured.out

    def test_list_runs_verbose(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        run_meta = SimpleNamespace(
            run_id="test-run-1",
            status=SimpleNamespace(value="completed"),
            total_samples=10,
            created_at="2026-01-01T00:00:00",
        )

        class FakeStorage:
            def __init__(self, root):
                pass

            def list_runs(self, limit=None):
                return [run_meta]

        monkeypatch.setattr("themis.storage.ExperimentStorage", FakeStorage)

        exit_code = list_command("runs", storage=str(storage_dir), verbose=True)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "test-run-1" in captured.out
        assert "completed" in captured.out
        assert "samples=10" in captured.out


# ---------------------------------------------------------------------------
# clean_command
# ---------------------------------------------------------------------------


class TestCleanCommand:
    """Tests for `themis clean` sub-command."""

    def test_clean_no_storage(self, tmp_path, capsys):
        nonexistent = tmp_path / "nope"
        exit_code = clean_command(storage=str(nonexistent), older_than=30)
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No storage found" in captured.out

    def test_clean_missing_older_than(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()
        exit_code = clean_command(storage=str(storage_dir))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "--older-than is required" in captured.out

    def test_clean_no_matching_runs(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        class FakeStorage:
            def __init__(self, root):
                pass

            def list_runs(self):
                return []

        monkeypatch.setattr("themis.storage.ExperimentStorage", FakeStorage)

        exit_code = clean_command(storage=str(storage_dir), older_than=30)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No runs matched" in captured.out

    def test_clean_dry_run(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        old_run = SimpleNamespace(
            run_id="old-run",
            created_at="2020-01-01T00:00:00",
        )

        class FakeStorage:
            def __init__(self, root):
                pass

            def list_runs(self):
                return [old_run]

            def delete_run(self, run_id):
                raise AssertionError("Should not delete in dry run")

        monkeypatch.setattr("themis.storage.ExperimentStorage", FakeStorage)

        exit_code = clean_command(storage=str(storage_dir), older_than=1, dry_run=True)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "old-run" in captured.out
        assert "Runs to delete" in captured.out

    def test_clean_actually_deletes(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        deleted = []
        old_run = SimpleNamespace(
            run_id="old-run",
            created_at="2020-01-01T00:00:00",
        )

        class FakeStorage:
            def __init__(self, root):
                pass

            def list_runs(self):
                return [old_run]

            def delete_run(self, run_id):
                deleted.append(run_id)

        monkeypatch.setattr("themis.storage.ExperimentStorage", FakeStorage)

        exit_code = clean_command(storage=str(storage_dir), older_than=1)
        assert exit_code == 0
        assert deleted == ["old-run"]
        captured = capsys.readouterr()
        assert "Deleted 1 run" in captured.out


# ---------------------------------------------------------------------------
# share_command
# ---------------------------------------------------------------------------


class TestShareCommand:
    """Tests for `themis share` sub-command."""

    def test_share_storage_not_found(self, tmp_path, capsys):
        nonexistent = tmp_path / "nope"
        exit_code = share_command("run-1", storage=str(nonexistent))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_share_success(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        fake_pack = SimpleNamespace(
            svg_path=tmp_path / "badge.svg",
            markdown_path=tmp_path / "results.md",
            markdown_snippet="![badge](badge.svg)",
            event_log_path=None,
        )

        def fake_create_share_pack(**kwargs):
            assert kwargs["run_id"] == "my-run"
            return fake_pack

        monkeypatch.setattr(
            "themis.experiment.share.create_share_pack",
            fake_create_share_pack,
        )

        exit_code = share_command("my-run", storage=str(storage_dir))
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Share assets created" in captured.out

    def test_share_file_not_found(self, tmp_path, capsys, monkeypatch):
        storage_dir = tmp_path / "experiments"
        storage_dir.mkdir()

        def fake_create_share_pack(**kwargs):
            raise FileNotFoundError("Run not found")

        monkeypatch.setattr(
            "themis.experiment.share.create_share_pack",
            fake_create_share_pack,
        )

        exit_code = share_command("missing-run", storage=str(storage_dir))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Run not found" in captured.err
