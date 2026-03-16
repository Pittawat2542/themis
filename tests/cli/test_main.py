from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    ProjectSpec,
    PromptTemplateSpec,
    SqliteBlobStorageSpec,
    TaskSpec,
)
from themis.cli.main import main
from themis.orchestration.run_manifest import RunManifest
from themis.storage.factory import build_storage_bundle
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.types.enums import DatasetSource


def _build_project(tmp_path: Path) -> ProjectSpec:
    return ProjectSpec(
        project_name="cli-report-demo",
        researcher_id="researcher",
        global_seed=13,
        storage=SqliteBlobStorageSpec(root_dir=str(tmp_path / "run-store")),
        execution_policy=ExecutionPolicySpec(),
    )


def _build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def _persisted_manifest_bundle(
    tmp_path: Path,
    *,
    run_id: str,
) -> tuple[Path, ProjectSpec, ExperimentSpec]:
    project = _build_project(tmp_path)
    experiment = _build_experiment()
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps(project.model_dump(mode="json")))
    storage_bundle = build_storage_bundle(project.storage)
    manifest = RunManifest(
        run_id=run_id,
        backend_kind="local",
        project_spec=project,
        experiment_spec=experiment,
    )
    RunManifestRepository(storage_bundle.manager).save_manifest(manifest)
    return project_path, project, experiment


class _StubParser:
    def __init__(
        self, *, parse_exit: object = None, handler_exit: object = None
    ) -> None:
        self._parse_exit = parse_exit
        self._handler_exit = handler_exit

    def parse_args(self, argv: list[str] | None) -> object:
        del argv
        if self._parse_exit is not None:
            raise SystemExit(self._parse_exit)

        def _handler(args: object) -> int:
            del args
            raise SystemExit(self._handler_exit)

        return SimpleNamespace(handler=_handler)

    def error(self, message: str) -> None:
        raise AssertionError(message)


def test_parent_cli_dispatches_quickcheck_help(capsys) -> None:
    assert main(["quickcheck", "--help"]) == 0
    assert "failures" in capsys.readouterr().out


def test_parent_cli_report_factory_writes_markdown(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"

    assert (
        main(
            [
                "report",
                "--factory",
                "tests.cli.fixture_factories:build_config_bundle",
                "--format",
                "markdown",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    content = output_path.read_text()
    assert "factory-report-demo" in content
    assert "<details>" in content
    assert "tests.cli.fixture_factories:build_config_bundle" in content
    assert "- Verbosity: default" in content


def test_parent_cli_report_run_manifest_loads_persisted_specs(
    tmp_path: Path, capsys
) -> None:
    project_path, _, _ = _persisted_manifest_bundle(tmp_path, run_id="run-report-1")

    assert (
        main(
            [
                "report",
                "--project-file",
                str(project_path),
                "--run-id",
                "run-report-1",
                "--format",
                "json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["header"]["entrypoint"] == "run_manifest:run-report-1"
    assert payload["header"]["project_name"] == "cli-report-demo"
    assert {child["name"] for child in payload["root"]["children"]} == {
        "project",
        "experiment",
    }


def test_parent_cli_report_run_manifest_reads_project_file(tmp_path: Path) -> None:
    project_path, _, _ = _persisted_manifest_bundle(tmp_path, run_id="run-report-2")

    output_path = tmp_path / "report.tex"
    assert (
        main(
            [
                "report",
                "--project-file",
                str(project_path),
                "--run-id",
                "run-report-2",
                "--format",
                "latex",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    assert "\\section*{Configuration Report}" in output_path.read_text()


def test_parent_cli_report_accepts_full_verbosity(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "report",
                "--factory",
                "tests.cli.fixture_factories:build_config_bundle",
                "--format",
                "json",
                "--verbosity",
                "full",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["header"]["verbosity"] == "full"


def test_parent_cli_dispatches_quickcheck_from_installed_entrypoint(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(sys, "argv", ["themis", "quickcheck", "--help"])

    assert main() == 0
    assert "failures" in capsys.readouterr().out


def test_parent_cli_dispatches_report_from_installed_entrypoint(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "themis",
            "report",
            "--factory",
            "tests.cli.fixture_factories:build_config_bundle",
            "--format",
            "json",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["header"]["project_name"] == "factory-report-demo"


def test_parent_cli_normalizes_system_exit_codes(monkeypatch) -> None:
    monkeypatch.setattr("themis.cli.main.build_parser", lambda: _StubParser())

    assert main(["ignored"]) == 0


def test_parent_cli_maps_string_system_exit_to_shell_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "themis.cli.main.build_parser",
        lambda: _StubParser(handler_exit="report generation failed"),
    )

    assert main(["ignored"]) == 1
