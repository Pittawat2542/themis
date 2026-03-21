"""CLI entry point for nested configuration reports."""

from __future__ import annotations

import importlib
from json import JSONDecodeError
from pathlib import Path
import tomllib
from typing import Literal, cast

from cyclopts import App
from rich.console import Console
from pydantic import ValidationError

from themis.cli._common import invoke_app
from themis.config_report import generate_config_report
from themis.errors import SpecValidationError
from themis.specs.experiment import ProjectSpec
from themis.storage.factory import build_storage_bundle
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.types.enums import ErrorCode
from themis.types.json_validation import format_validation_error


def build_app(*, standalone: bool = False) -> App:
    """Build the config-report Cyclopts app."""

    app = App(
        name="themis-report" if standalone else "report",
        help="Render reproducibility-friendly config reports from a factory or run.",
    )

    @app.default
    def report(
        factory: str | None = None,
        project_file: str | None = None,
        run_id: str | None = None,
        format: str = "markdown",
        verbosity: str = "default",
        output: str | None = None,
    ) -> int:
        if (factory is None) == (project_file is None):
            return _emit_error("Pass exactly one of --factory or --project-file.")
        if format not in {"json", "yaml", "markdown", "latex"}:
            return _emit_error("--format must be one of: json, yaml, markdown, latex.")
        if verbosity not in {"default", "full"}:
            return _emit_error("--verbosity must be one of: default, full.")
        resolved_format = cast(
            Literal["json", "yaml", "markdown", "latex"],
            format,
        )
        resolved_verbosity = cast(Literal["default", "full"], verbosity)
        if factory is not None:
            if run_id is not None:
                return _emit_error("--run-id requires --project-file.")
            config = _load_factory(factory)
            entrypoint = factory
        else:
            if run_id is None:
                return _emit_error("--run-id is required with --project-file.")
            assert project_file is not None
            config = _load_run_manifest_bundle(project_file, run_id)
            entrypoint = f"run_manifest:{run_id}"

        rendered = generate_config_report(
            config,
            format=resolved_format,
            output=output,
            entrypoint=entrypoint,
            verbosity=resolved_verbosity,
        )
        if output is None:
            print(rendered, end="")
        return 0

    return app


def _load_factory(factory_path: str) -> object:
    module_name, _, attr_name = factory_path.partition(":")
    if not module_name or not attr_name:
        raise ValueError("Factory must use the form module:function.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"{factory_path} is not callable.")
    return factory()


def _load_project_spec(path: str) -> ProjectSpec:
    project_path = Path(path)
    try:
        if project_path.suffix == ".toml":
            with project_path.open("rb") as fh:
                payload = tomllib.load(fh)
            return ProjectSpec.model_validate(payload)
        if project_path.suffix == ".json":
            return ProjectSpec.model_validate_json(project_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=f"Failed to parse project config {project_path.name}: {exc}",
        ) from exc
    except JSONDecodeError as exc:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=f"Failed to parse project config {project_path.name}: {exc}",
        ) from exc
    except ValidationError as exc:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=(
                f"Failed to parse project config {project_path.name}: "
                f"{format_validation_error(exc)}"
            ),
        ) from exc
    raise SpecValidationError(
        code=ErrorCode.SCHEMA_MISMATCH,
        message="Project files must use .toml or .json.",
    )


def _load_run_manifest_bundle(project_file: str, run_id: str) -> dict[str, object]:
    project = _load_project_spec(project_file)
    storage_bundle = build_storage_bundle(project.storage)
    manifest = RunManifestRepository(storage_bundle.manager).get_manifest(run_id)
    if manifest is None:
        raise ValueError(f"Run manifest '{run_id}' was not found.")
    bundle: dict[str, object] = {"project": manifest.project_spec or project}
    if manifest.benchmark_spec is not None:
        bundle["benchmark"] = manifest.benchmark_spec
    else:
        bundle["experiment"] = manifest.experiment_spec
    return bundle


def main(argv: list[str] | None = None) -> int:
    """Run the config-report CLI."""

    return invoke_app(build_app(standalone=True), argv)


def _emit_error(message: str) -> int:
    Console(stderr=True, markup=False).print(message)
    return 1
