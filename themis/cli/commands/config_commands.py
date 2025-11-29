"""Configuration-related commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

from themis.cli.utils import effective_total, export_outputs
from themis.config import (
    load_dataset_from_config,
    load_experiment_config,
    run_experiment_from_config,
    summarize_report_for_config,
)
from themis.utils.logging_utils import configure_logging
from themis.utils.progress import ProgressReporter


def run_configured_experiment(
    *,
    config: Annotated[
        Path, Parameter(help="Path to a Hydra/OmegaConf experiment config file")
    ],
    overrides: Annotated[
        tuple[str, ...],
        Parameter(
            help="Optional Hydra-style overrides (e.g. generation.sampling.temperature=0.2)",
            show_default=False,
        ),
    ] = (),
    log_level: Annotated[
        str, Parameter(help="Logging level (critical/error/warning/info/debug/trace)")
    ] = "info",
    csv_output: Annotated[
        Path | None, Parameter(help="Write CSV export to this path")
    ] = None,
    html_output: Annotated[
        Path | None, Parameter(help="Write HTML summary to this path")
    ] = None,
    json_output: Annotated[
        Path | None, Parameter(help="Write JSON export to this path")
    ] = None,
) -> int:
    """Execute an experiment described via config file."""
    configure_logging(log_level)
    experiment_config = load_experiment_config(config, overrides)
    dataset = load_dataset_from_config(experiment_config)
    total = effective_total(len(dataset), experiment_config.max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = run_experiment_from_config(
            experiment_config,
            dataset=dataset,
            on_result=progress.on_result,
        )
    print(summarize_report_for_config(experiment_config, report))
    export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title=f"{experiment_config.name} experiment",
    )
    return 0


def validate_config(
    *,
    config: Annotated[Path, Parameter(help="Path to config file to validate")],
) -> int:
    """Validate a configuration file without running the experiment."""
    if not config.exists():
        print(f"❌ Error: Config file not found: {config}")
        return 1

    print(f"Validating config: {config}")
    print("-" * 60)

    try:
        # Try to load as experiment config
        experiment_config = load_experiment_config(config, overrides=())
        print("✓ Config file is valid")
        print(f"\nExperiment: {experiment_config.name}")
        print(f"Run ID: {experiment_config.run_id or '(auto-generated)'}")
        print(f"Resume: {experiment_config.resume}")
        print(f"Max samples: {experiment_config.max_samples or '(unlimited)'}")

        print("\nDataset:")
        print(f"  Source: {experiment_config.dataset.source}")
        print(f"  Split: {experiment_config.dataset.split}")
        if experiment_config.dataset.limit:
            print(f"  Limit: {experiment_config.dataset.limit}")
        if experiment_config.dataset.subjects:
            print(f"  Subjects: {', '.join(experiment_config.dataset.subjects)}")

        print("\nGeneration:")
        print(f"  Model: {experiment_config.generation.model_identifier}")
        print(f"  Provider: {experiment_config.generation.provider.name}")
        print(f"  Temperature: {experiment_config.generation.sampling.temperature}")
        print(f"  Max tokens: {experiment_config.generation.sampling.max_tokens}")

        if experiment_config.storage.path:
            print(f"\nStorage: {experiment_config.storage.path}")

        return 0
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return 1


def init_config(
    *,
    output: Annotated[Path, Parameter(help="Output path for config file")] = Path(
        "themis_config.yaml"
    ),
    template: Annotated[
        Literal["basic", "math500", "inline"],
        Parameter(help="Config template to generate"),
    ] = "basic",
) -> int:
    """Generate a sample configuration file for use with run-config."""
    templates = {
        "basic": """name: my_experiment
task: math500
dataset:
  source: huggingface
  dataset_id: math500
  limit: 50
generation:
  model_identifier: fake-math-llm
  provider:
    name: fake
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
  runner:
    max_parallel: 1
    max_retries: 3
storage:
  path: .cache/my_experiment
run_id: my-experiment-001
resume: true
""",
        "math500": """name: math500_evaluation
task: math500
dataset:
  source: huggingface
  dataset_id: math500
  limit: null  # No limit, run full dataset
  subjects:
    - algebra
    - geometry
generation:
  model_identifier: my-model
  provider:
    name: openai-compatible
    options:
      base_url: http://localhost:1234/v1
      api_key: not-needed
      model_name: qwen2.5-7b-instruct
      timeout: 60
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
  runner:
    max_parallel: 4
    max_retries: 3
    retry_initial_delay: 0.5
    retry_backoff_multiplier: 2.0
    retry_max_delay: 2.0
storage:
  path: .cache/math500
run_id: math500-run-001
resume: true
max_samples: null
""",
        "inline": """name: inline_dataset_experiment
task: math500
dataset:
  source: inline
  inline_samples:
    - unique_id: sample-1
      problem: "What is 2 + 2?"
      answer: "4"
      subject: arithmetic
      level: 1
    - unique_id: sample-2
      problem: "Solve for x: 2x + 5 = 13"
      answer: "4"
      subject: algebra
      level: 2
generation:
  model_identifier: fake-math-llm
  provider:
    name: fake
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
storage:
  path: .cache/inline_experiment
run_id: inline-001
resume: true
""",
    }

    if output.exists():
        print(f"❌ Error: File already exists: {output}")
        print("   Use a different --output path or delete the existing file")
        return 1

    config_content = templates[template]

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(config_content)

        print(f"✓ Created config file: {output}")
        print(f"  Template: {template}")
        print("\n📝 Next steps:")
        print(f"  1. Edit {output} to customize settings")
        print(
            f"  2. Validate: uv run python -m themis.cli validate-config --config {output}"
        )
        print(f"  3. Run: uv run python -m themis.cli run-config --config {output}")

        if template == "math500":
            print("\n⚠️  Remember to:")
            print("  • Update provider.options.base_url with your LLM server endpoint")
            print("  • Update provider.options.model_name with your actual model")
            print("  • Set provider.options.api_key if required by your server")
        elif template == "inline":
            print("\n💡 Tip:")
            print("  • Add more samples to dataset.inline_samples list")
            print("  • Each sample needs: unique_id, problem, answer")

        return 0
    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return 1
