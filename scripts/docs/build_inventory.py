from __future__ import annotations

import json
import tomllib
from pathlib import Path

import themis.catalog
import themis
from themis.cli.app import app


REPO_ROOT = Path(__file__).resolve().parents[2]


def _cli_commands() -> list[str]:
    root_commands = sorted(
        name for name in app.resolved_commands() if not name.startswith("-")
    )
    commands: list[str] = []
    for command in root_commands:
        commands.append(command)
        if command == "quick-eval":
            commands.extend(
                f"{command} {sub}"
                for sub in ("inline", "file", "huggingface", "benchmark")
            )
        elif command == "export":
            commands.extend(f"{command} {sub}" for sub in ("generation", "evaluation"))
        elif command == "worker":
            commands.append("worker run")
        elif command == "batch":
            commands.append("batch run")
    return commands


def _manifest_table(path: Path, key: str) -> list[str]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return sorted(payload[key])


def main() -> None:
    payload = {
        "public_exports": sorted(themis.__all__),
        "catalog_exports": sorted(themis.catalog.__all__),
        "cli_commands": _cli_commands(),
        "builtin_components": _manifest_table(
            REPO_ROOT / "themis" / "catalog" / "manifests" / "components.toml",
            "components",
        ),
        "benchmarks": _manifest_table(
            REPO_ROOT
            / "themis"
            / "catalog"
            / "benchmarks"
            / "manifests"
            / "benchmarks.toml",
            "benchmarks",
        ),
        "docs_destinations": {
            "home": "docs/index.md",
            "start-here": "docs/start-here/index.md",
            "tutorials": "docs/tutorials",
            "how-to": "docs/how-to",
            "reference": "docs/reference/index.md",
            "explanation": "docs/explanation/index.md",
            "glossary": "docs/glossary.md",
            "faq": "docs/faq.md",
            "project": "docs/project/index.md",
        },
        "required_topics": {
            "catalog_api": [
                "themis.catalog.load(...)",
                "themis.catalog.run(...)",
            ],
            "observability": [
                "LifecycleSubscriber",
                "TracingProvider",
            ],
            "config_loading": [
                "Experiment.from_config(...)",
                "YAML",
                "TOML",
                "overrides",
                "relative to the config file directory",
            ],
            "stage_execution": [
                "until_stage",
                "completed_through_stage",
                "existing_run_policy",
                "Python-only",
            ],
            "prompt_and_estimates": [
                "PromptSpec.blocks",
                "estimated_total_tokens",
                "assumptions",
            ],
            "reporting_shape": [
                "error_category",
                "error_message",
                "outcome_counts",
                "error_counts",
            ],
            "limitations": [
                "no native provider batch API orchestration",
                "no config diff tooling",
                "no first-class grid-search reuse",
                "no storage-efficiency redesign for very large artifacts",
                "cross-environment reproducibility still relies on user-managed dependency locking",
            ],
            "cache_behavior": [
                "stage caches are keyed by stage inputs, not by `run_id`",
                "persistent stores are required for cross-run cache reuse",
                "`InMemoryRunStore` does not provide cross-run stage cache behavior",
            ],
        },
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
