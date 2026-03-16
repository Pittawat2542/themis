"""CLI entry point for nested configuration reports."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import tomllib
from typing import Any

from pydantic import ValidationError

from themis.config_report import generate_config_report
from themis.errors import SpecValidationError
from themis.specs.experiment import ProjectSpec
from themis.storage.factory import build_storage_bundle
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.types.enums import ErrorCode
from themis.types.json_validation import format_validation_error


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach config-report arguments to an argparse parser."""

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--factory")
    input_group.add_argument("--project-file")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "markdown", "latex"],
        default="markdown",
    )
    parser.add_argument(
        "--verbosity",
        choices=["default", "full"],
        default="default",
    )
    parser.add_argument("--output")


def configure_report_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure one parser to serve the config-report CLI."""

    add_report_arguments(parser)
    parser.set_defaults(handler=run_with_args, _parser=parser)
    return parser


def build_parser(*, prog: str = "themis report") -> argparse.ArgumentParser:
    """Build the config-report CLI parser."""

    parser = argparse.ArgumentParser(prog=prog)
    return configure_report_parser(parser)


def add_report_subparser(
    subparsers: argparse._SubParsersAction[Any],
) -> argparse.ArgumentParser:
    """Add the report command to a parent CLI."""

    parser = subparsers.add_parser("report")
    return configure_report_parser(parser)


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
    except ValidationError as exc:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=(
                f"Failed to parse project config {project_path.name}: "
                f"{format_validation_error(exc)}"
            ),
        ) from exc
    raise ValueError("Project files must use .toml or .json.")


def _load_run_manifest_bundle(project_file: str, run_id: str) -> dict[str, object]:
    project = _load_project_spec(project_file)
    storage_bundle = build_storage_bundle(project.storage)
    manifest = RunManifestRepository(storage_bundle.manager).get_manifest(run_id)
    if manifest is None:
        raise ValueError(f"Run manifest '{run_id}' was not found.")
    return {
        "project": manifest.project_spec or project,
        "experiment": manifest.experiment_spec,
    }


def run_with_args(args: argparse.Namespace) -> int:
    """Execute the parsed config-report command."""

    parser: argparse.ArgumentParser = args._parser
    if args.factory is not None:
        if args.run_id is not None:
            parser.error("--run-id requires --project-file.")
        config = _load_factory(args.factory)
        entrypoint = args.factory
    else:
        if args.run_id is None:
            parser.error("--run-id is required with --project-file.")
        assert args.project_file is not None
        config = _load_run_manifest_bundle(args.project_file, args.run_id)
        entrypoint = f"run_manifest:{args.run_id}"

    rendered = generate_config_report(
        config,
        format=args.format,
        output=args.output,
        entrypoint=entrypoint,
        verbosity=args.verbosity,
    )
    if args.output is None:
        print(rendered, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the config-report CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_with_args(args)
