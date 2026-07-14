"""Suite CLI commands."""

from __future__ import annotations

from cyclopts import App

from themis.catalog import expand_suite, get_suite, list_suites, run_suite
from themis.cli.helpers import dump_json

suite_app = App(name="suite", help="Benchmark suite workflows.")


@suite_app.command(name="list")
def list_command() -> int:
    """List available suite definitions."""

    print("\n".join(list_suites()))
    return 0


@suite_app.command
def inspect(suite_id: str) -> int:
    """Inspect one expanded suite definition as JSON."""

    suite = get_suite(suite_id)
    expansion = expand_suite(suite_id)
    print(
        dump_json(
            "suite.inspect",
            {
                "suite_id": suite.suite_id,
                "aggregation": suite.aggregation.value,
                "description": suite.description,
                "tags": suite.tags,
                "items": [item.model_dump(mode="json") for item in expansion.items],
            },
        )
    )
    return 0


@suite_app.command
def run(suite_id: str) -> int:
    """Run a suite with the default in-memory store."""

    result = run_suite(suite_id)
    payload = result.model_dump(mode="json")
    payload["run_ids"] = result.run_ids
    print(dump_json("suite.run", payload))
    return 0
