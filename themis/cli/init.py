"""Project scaffold generator for Themis starter projects."""

from __future__ import annotations

import json
from pathlib import Path
import re
from textwrap import dedent

from cyclopts import App
from rich.console import Console


def build_app() -> App:
    """Build the project-init Cyclopts app."""

    app = App(
        name="init",
        help="Generate a lean benchmark-first starter project.",
    )

    @app.default
    def init(
        path: str,
        template: str = "qa",
        provider: str = "demo",
        model: str | None = None,
    ) -> int:
        try:
            project_root = Path(path)
            package_name = _package_name(project_root.name)
            normalized_provider = provider.replace("-", "_")
            if template not in {"qa", "mcq"}:
                raise ValueError("--template must be one of: qa, mcq.")
            if normalized_provider not in {"demo", "openai", "openai_compatible"}:
                raise ValueError(
                    "--provider must be one of: demo, openai, openai-compatible."
                )
            if project_root.exists() and any(project_root.iterdir()):
                raise ValueError(
                    f"Refusing to scaffold into non-empty directory {project_root}."
                )
            project_root.mkdir(parents=True, exist_ok=True)
            files = _build_scaffold_files(
                project_root=project_root,
                package_name=package_name,
                template=template,
                provider=normalized_provider,
                model=model or _default_model(normalized_provider),
            )
            for destination, content in files.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
            Console().print(f"Initialized starter project at {project_root}")
            return 0
        except Exception as exc:
            Console(stderr=True).print(str(exc))
            return 1

    return app


def _build_scaffold_files(
    *,
    project_root: Path,
    package_name: str,
    template: str,
    provider: str,
    model: str,
) -> dict[Path, str]:
    storage_dir = f".cache/{package_name}"
    metric_name = "choice_accuracy" if template == "mcq" else "normalized_exact_match"
    extractor_name = "choice_letter" if template == "mcq" else "normalized_text"
    prompt = (
        "{item.input}"
        if template == "qa"
        else "Answer with the best option letter only.\\n\\n{item.input}"
    )
    dataset_rows = (
        [
            {
                "item_id": "sample-1",
                "input": "Question: What is 2 + 2?",
                "expected": "4",
            }
        ]
        if template == "qa"
        else [
            {
                "item_id": "sample-1",
                "input": "Question: Which letter comes first alphabetically?\\nA. C\\nB. A\\nC. B\\nD. D",
                "expected": "B",
            }
        ]
    )
    sample_jsonl = "\n".join(json.dumps(row) for row in dataset_rows) + "\n"
    files = {
        project_root / "project.toml": dedent(
            f"""
            project_name = "{package_name}"
            researcher_id = "themis"
            global_seed = 7

            [storage]
            root_dir = "{storage_dir}"
            backend = "sqlite_blob"
            store_item_payloads = true
            compression = "none"

            [execution_policy]
            max_retries = 2
            retry_backoff_factor = 1.5
            circuit_breaker_threshold = 4
            """
        ).lstrip(),
        project_root / ".env.example": dedent(
            f"""
            THEMIS_STARTER_PROVIDER={provider}
            THEMIS_STARTER_MODEL={model}
            OPENAI_API_KEY=
            OPENAI_COMPAT_API_KEY=
            OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8000/v1
            """
        ).lstrip(),
        project_root / "README.md": _readme_template(
            package_name=package_name,
            metric_name=metric_name,
            provider=provider,
            model=model,
            storage_dir=storage_dir,
        ),
        project_root / "data" / "sample.jsonl": sample_jsonl,
        project_root
        / package_name
        / "__init__.py": '"""Starter benchmark package."""\n',
        project_root / package_name / "__main__.py": dedent(
            f"""
            from {package_name}.app import main


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).lstrip(),
        project_root / package_name / "settings.py": _settings_template(
            provider=provider,
            model=model,
        ),
        project_root / package_name / "registry.py": dedent(
            """
            from themis.cli.starter_catalog import build_starter_registry

            from .settings import get_settings


            def build_registry():
                settings = get_settings()
                return build_starter_registry(settings.provider_name)
            """
        ).lstrip(),
        project_root / package_name / "benchmarks" / "__init__.py": dedent(
            """
            from .default import build_benchmark

            __all__ = ["build_benchmark"]
            """
        ).lstrip(),
        project_root / package_name / "benchmarks" / "default.py": _benchmark_template(
            prompt=prompt,
            metric_name=metric_name,
            extractor_name=extractor_name,
        ),
        project_root / package_name / "datasets" / "__init__.py": dedent(
            """
            from .local_file import build_dataset_provider

            __all__ = ["build_dataset_provider"]
            """
        ).lstrip(),
        project_root / package_name / "datasets" / "local_file.py": dedent(
            """
            from themis.cli.starter_catalog import StarterDatasetProvider


            def build_dataset_provider():
                return StarterDatasetProvider()
            """
        ).lstrip(),
        project_root / package_name / "app.py": _app_template(
            package_name=package_name
        ),
    }
    return files


def _settings_template(*, provider: str, model: str) -> str:
    return dedent(
        f"""
        from __future__ import annotations

        import os
        from dataclasses import dataclass


        @dataclass(frozen=True)
        class StarterSettings:
            provider: str = os.getenv("THEMIS_STARTER_PROVIDER", "{provider}")
            model_id: str = os.getenv("THEMIS_STARTER_MODEL", "{model}")
            openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
            openai_compat_api_key: str | None = os.getenv("OPENAI_COMPAT_API_KEY")
            openai_compat_base_url: str = os.getenv(
                "OPENAI_COMPAT_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ).rstrip("/")

            @property
            def provider_name(self) -> str:
                return self.provider.replace("-", "_")

            def model_extras(self) -> dict[str, object]:
                if self.provider_name == "openai_compatible":
                    extras: dict[str, object] = {{
                        "base_url": self.openai_compat_base_url,
                        "timeout_seconds": 60.0,
                    }}
                    if self.openai_compat_api_key:
                        extras["api_key"] = self.openai_compat_api_key
                    return extras
                if self.provider_name == "openai" and self.openai_api_key:
                    return {{"api_key": self.openai_api_key}}
                return {{}}


        def get_settings() -> StarterSettings:
            return StarterSettings()
        """
    ).lstrip()


def _benchmark_template(
    *,
    prompt: str,
    metric_name: str,
    extractor_name: str,
) -> str:
    return dedent(
        f"""
        from pathlib import Path

        from themis import (
            BenchmarkSpec,
            InferenceGridSpec,
            InferenceParamsSpec,
            ModelSpec,
            ParseSpec,
            PromptMessage,
            PromptVariantSpec,
            ScoreSpec,
            SliceSpec,
        )
        from themis.specs import DatasetSpec, GenerationSpec
        from themis.types.enums import DatasetSource, PromptRole


        def build_benchmark(settings) -> BenchmarkSpec:
            return BenchmarkSpec(
                benchmark_id="starter-benchmark",
                models=[
                    ModelSpec(
                        model_id=settings.model_id,
                        provider=settings.provider_name,
                        extras=settings.model_extras(),
                    )
                ],
                slices=[
                    SliceSpec(
                        slice_id="starter-slice",
                        dataset=DatasetSpec(
                            source=DatasetSource.LOCAL,
                            dataset_id=str(Path("data/sample.jsonl")),
                        ),
                        prompt_variant_ids=["starter-default"],
                        generation=GenerationSpec(),
                        parses=[ParseSpec(name="parsed", extractors=["{extractor_name}"])],
                        scores=[
                            ScoreSpec(
                                name="default",
                                parse="parsed",
                                metrics=["{metric_name}"],
                            )
                        ],
                    )
                ],
                prompt_variants=[
                    PromptVariantSpec(
                        id="starter-default",
                        family="starter",
                        messages=[
                            PromptMessage(role=PromptRole.USER, content="{prompt}")
                        ],
                    )
                ],
                inference_grid=InferenceGridSpec(
                    params=[InferenceParamsSpec(max_tokens=128, temperature=0.0)]
                ),
            )
        """
    ).lstrip()


def _app_template(*, package_name: str) -> str:
    return dedent(
        f"""
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path
        import sys
        import tomllib

        from rich.console import Console
        from rich.table import Table

        from themis import Orchestrator, ProjectSpec
        from themis.progress import (
            ProgressConfig,
            ProgressRendererType,
            ProgressVerbosity,
        )
        from themis.cli.starter_catalog import load_local_rows

        from {package_name}.benchmarks import build_benchmark
        from {package_name}.datasets import build_dataset_provider
        from {package_name}.registry import build_registry
        from {package_name}.settings import get_settings


        def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
            parser = argparse.ArgumentParser(description="Run the starter Themis benchmark.")
            parser.add_argument("--preview", action="store_true")
            parser.add_argument("--estimate-only", action="store_true")
            parser.add_argument("--format", choices=("table", "json"), default="table")
            return parser.parse_args(argv)


        def _load_project_spec() -> ProjectSpec:
            with Path("project.toml").open("rb") as fh:
                return ProjectSpec.model_validate(tomllib.load(fh))


        def _build_progress_config() -> ProgressConfig:
            renderer = (
                ProgressRendererType.RICH
                if sys.stderr.isatty()
                else ProgressRendererType.LOG
            )
            return ProgressConfig(
                enabled=True,
                renderer=renderer,
                verbosity=ProgressVerbosity.NORMAL,
            )


        def build_report_bundle() -> dict[str, object]:
            settings = get_settings()
            return {{
                "project": _load_project_spec(),
                "benchmark": build_benchmark(settings),
            }}


        def main(argv: list[str] | None = None) -> int:
            args = _parse_args(argv)
            settings = get_settings()
            project = _load_project_spec()
            benchmark = build_benchmark(settings)

            if args.preview:
                sample = load_local_rows(Path("data/sample.jsonl"))[0]
                payload = benchmark.preview(sample)
                if args.format == "json":
                    print(json.dumps(payload, indent=2))
                else:
                    console = Console()
                    table = Table(title="Preview")
                    table.add_column("Prompt Variant")
                    table.add_column("Messages")
                    for entry in payload:
                        messages = "\\n".join(
                            f"[{{message['role']}}] {{message['content']}}"
                            for message in entry["messages"]
                        )
                        table.add_row(entry["prompt_variant_id"], messages)
                    console.print(table)
                return 0

            orchestrator = Orchestrator.from_project_file(
                "project.toml",
                registry=build_registry(),
                dataset_provider=build_dataset_provider(),
            )

            if args.estimate_only:
                estimate = orchestrator.estimate(benchmark)
                if args.format == "json":
                    print(json.dumps(estimate.model_dump(mode="json"), indent=2))
                else:
                    console = Console()
                    table = Table(title="Estimate")
                    table.add_column("Trial Count")
                    table.add_column("Total Work Items")
                    table.add_row(
                        str(estimate.trial_count),
                        str(estimate.total_work_items),
                    )
                    console.print(table)
                return 0

            result = orchestrator.run_benchmark(
                benchmark,
                progress=_build_progress_config(),
            )
            rows = result.aggregate(
                group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
            )
            if args.format == "json":
                print(json.dumps(rows, indent=2))
            else:
                console = Console()
                table = Table(title="Results")
                table.add_column("Model")
                table.add_column("Slice")
                table.add_column("Metric")
                table.add_column("Mean")
                table.add_column("Count")
                for row in rows:
                    table.add_row(
                        str(row["model_id"]),
                        str(row["slice_id"]),
                        str(row["metric_id"]),
                        str(row["mean"]),
                        str(row["count"]),
                    )
                console.print(table)
                console.print(f"SQLite DB: {{project.storage.root_dir}}/themis.sqlite3")
            return 0
        """
    ).lstrip()


def _readme_template(
    *,
    package_name: str,
    metric_name: str,
    provider: str,
    model: str,
    storage_dir: str,
) -> str:
    return dedent(
        f"""
        # {package_name}

        Starter benchmark scaffold generated by `themis init`.

        ## Defaults

        - provider: `{provider}`
        - model: `{model}`
        - metric: `{metric_name}`

        ## Preview

        ```bash
        uv run python -m {package_name} --preview
        ```

        ## Run

        ```bash
        uv run python -m {package_name}
        ```

        ## Inspect Stored Results

        ```bash
        themis quickcheck scores --db {storage_dir}/themis.sqlite3 --metric {metric_name}
        ```

        ## Render a Config Report

        ```bash
        themis report --factory {package_name}.app:build_report_bundle --format markdown
        ```
        """
    ).lstrip()


def _package_name(raw_name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip().lower())
    if not value:
        return "themis_starter"
    if value[0].isdigit():
        return f"project_{value}"
    return value


def _default_model(provider: str) -> str:
    if provider == "demo":
        return "demo-model"
    return "gpt-4o-mini"
