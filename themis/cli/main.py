"""Cyclopts-powered CLI entrypoints for Themis."""

from __future__ import annotations

from typing import Sequence

from cyclopts import App

# Import command modules
from themis.cli.commands import (
    benchmarks,
    comparison,
    config_commands,
    cost,
    demo,
    info,
    leaderboard,
    sample_run,
    visualize,
)
from themis.cli.commands import math_benchmarks as math_cmds
from themis.cli.commands import mcq_benchmarks as mcq_cmds

# Import provider modules to ensure they register themselves
try:
    from themis.generation import clients  # noqa: F401 - registers fake provider
    from themis.generation.providers import (
        litellm_provider,  # noqa: F401
        vllm_provider,  # noqa: F401
    )
except ImportError:
    pass  # Some providers may not be available

app = App(help="Run Themis experiments from the command line")

# Register demo command
app.command(name="demo")(demo.demo_command)

# Register math benchmark commands
app.command(name="math500")(math_cmds.math500_command)
app.command(name="aime24")(math_cmds.aime24_command)
app.command(name="aime25")(math_cmds.aime25_command)
app.command(name="amc23")(math_cmds.amc23_command)
app.command(name="olympiadbench")(math_cmds.olympiadbench_command)
app.command(name="beyondaime")(math_cmds.beyond_aime_command)

# Register MCQ benchmark commands
app.command(name="supergpqa")(mcq_cmds.supergpqa_command)
app.command(name="mmlu-pro")(mcq_cmds.mmlu_pro_command)

# Register config commands
app.command(name="run-config")(config_commands.run_configured_experiment)
app.command(name="validate-config")(config_commands.validate_config)
app.command(name="init")(config_commands.init_config)

# Register info and listing commands
app.command(name="list-providers")(benchmarks.list_providers)
app.command(name="list-benchmarks")(benchmarks.list_benchmarks)
app.command(name="info")(info.show_info)
app.command(name="new-project")(info.new_project)

# Register comparison commands
app.command(name="compare")(comparison.compare_command)
app.command(name="diff")(comparison.diff_command)
app.command(name="pareto")(comparison.pareto_command)

# Register cost commands
app.command(name="estimate-cost")(cost.estimate_cost_command)
app.command(name="show-pricing")(cost.show_pricing_command)

# Register visualization commands
app.command(name="visualize")(visualize.visualize_comparison_command)
app.command(name="visualize-pareto")(visualize.visualize_pareto_command)
app.command(name="visualize-distribution")(visualize.visualize_distribution_command)

# Register leaderboard command
app.command(name="leaderboard")(leaderboard.leaderboard_command)

# Register sample-run command
app.command(name="sample-run")(sample_run.sample_run_command)


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(argv) if argv is not None else None
    try:
        result = app(parsed_argv)
    except SystemExit as exc:  # pragma: no cover - CLI integration path
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
