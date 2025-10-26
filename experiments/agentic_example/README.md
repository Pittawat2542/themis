# Agentic Example

This example shows how to plug in a custom generation runner that performs a
plan → answer loop while still leaning on Themis abstractions.

- `config.py` extends the base config with `planner_prompt` and
  `final_prompt_prefix` so you can tweak the agent behavior without touching the
  runner.
- `runner.py` defines `AgenticRunner`, which calls the provider twice: first to
  draft a plan, then to produce the final answer using that plan.
- `experiment.py` wires the builder with multi-metric evaluation (`ExactMatch`
  and `ResponseLength`) to demonstrate scoring per record with multiple metrics.
- `cli.py` exposes a Cyclopts CLI (`python -m experiments.agentic_example.cli run`).

Try it:

```bash
uv run python -m experiments.agentic_example.cli run --dry-run
```
