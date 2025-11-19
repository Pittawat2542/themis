"""Sample run command for quick testing before full experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.cli.commands.config_commands import run_configured_experiment


def sample_run_command(
    *,
    config: Annotated[Path, Parameter(help="Path to experiment configuration file")],
    n: Annotated[int, Parameter(help="Number of samples to test")] = 5,
    verbose: Annotated[bool, Parameter(help="Show detailed output")] = False,
    show_outputs: Annotated[
        bool, Parameter(help="Display sample outputs and predictions")
    ] = False,
    estimate_cost: Annotated[
        bool, Parameter(help="Estimate full run cost based on sample")
    ] = True,
) -> int:
    """Quick test run on N samples before running full experiment.

    This command helps you:
    - Test your configuration works correctly
    - Preview sample outputs before full run
    - Estimate total cost based on actual token usage
    - Catch configuration errors early
    - Iterate on prompts quickly

    Examples:
        # Basic quick test
        uv run python -m themis.cli sample-run \\
          --config my_config.yaml \\
          --n 5

        # Test with verbose output
        uv run python -m themis.cli sample-run \\
          --config my_config.yaml \\
          --n 3 \\
          --verbose \\
          --show-outputs

        # Test and estimate full run cost
        uv run python -m themis.cli sample-run \\
          --config my_config.yaml \\
          --n 10 \\
          --estimate-cost
    """
    try:
        import json
        import tempfile

        from hydra import compose, initialize_config_dir

        # Load config
        config_path = Path(config).resolve()
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return 1

        config_dir = str(config_path.parent)
        config_name = config_path.stem

        print("=" * 80)
        print(f"🧪 Sample Run: Testing {n} samples")
        print("=" * 80)
        print(f"Config: {config_path}")
        print(f"Samples: {n}")
        print()

        # Initialize Hydra
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)

            # Override dataset limit
            original_limit = cfg.dataset.get("limit")
            cfg.dataset.limit = n

            # Use temporary storage
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg.storage.path = temp_dir

                # Generate temporary run_id
                cfg.run_id = "sample-run-temp"
                cfg.resume = False

                print("📋 Configuration:")
                print(f"  Model: {cfg.generation.model_identifier}")
                print(f"  Provider: {cfg.generation.provider.name}")
                print(f"  Temperature: {cfg.generation.sampling.temperature}")
                print(f"  Max tokens: {cfg.generation.sampling.max_tokens}")
                if hasattr(cfg.dataset, "source"):
                    print(f"  Dataset: {cfg.dataset.source}")
                print()

                # Run experiment on sample
                print("🚀 Running sample experiment...")
                print()

                # Redirect to capture run
                result = run_configured_experiment(
                    config_path=config_path,
                    overrides=[
                        f"dataset.limit={n}",
                        f"storage.path={temp_dir}",
                        "run_id=sample-run-temp",
                        "resume=false",
                    ],
                )

                if result != 0:
                    print("\n❌ Sample run failed")
                    return result

                # Load results
                report_path = Path(temp_dir) / "sample-run-temp" / "report.json"
                if not report_path.exists():
                    print("\n⚠️  No report generated")
                    return 1

                with report_path.open("r") as f:
                    report_data = json.load(f)

                # Display results
                print("\n" + "=" * 80)
                print("✅ Sample Run Complete")
                print("=" * 80)

                # Metrics
                metrics = report_data.get("metrics", [])
                if metrics:
                    print("\n📊 Metrics:")
                    for metric in metrics:
                        name = metric["name"]
                        mean = metric["mean"]
                        count = metric["count"]
                        print(f"  {name}: {mean:.4f} (n={count})")

                # Cost analysis
                cost_data = report_data.get("summary", {}).get("cost")
                if cost_data:
                    total_cost = cost_data.get("total_cost", 0)
                    token_counts = cost_data.get("token_counts", {})
                    prompt_tokens = token_counts.get("prompt_tokens", 0)
                    completion_tokens = token_counts.get("completion_tokens", 0)

                    print("\n💰 Cost (sample run):")
                    print(f"  Total: ${total_cost:.4f}")
                    print(f"  Per sample: ${total_cost / n:.6f}")
                    print(
                        f"  Prompt tokens: {prompt_tokens} ({prompt_tokens / n:.0f} avg)"
                    )
                    print(
                        f"  Completion tokens: {completion_tokens} ({completion_tokens / n:.0f} avg)"
                    )

                    # Estimate full run cost
                    if estimate_cost and original_limit:
                        full_cost = (total_cost / n) * original_limit
                        print("\n📈 Estimated full run cost:")
                        print(f"  Dataset size: {original_limit} samples")
                        print(f"  Estimated cost: ${full_cost:.2f}")
                        print(
                            f"  95% CI: ${full_cost * 0.8:.2f} - ${full_cost * 1.2:.2f}"
                        )

                        if full_cost > 10.0:
                            print(f"\n⚠️  Warning: Estimated cost is ${full_cost:.2f}")
                            print("  Consider using --limit for initial testing")

                # Failures
                failures = report_data.get("run_failures", [])
                eval_failures = report_data.get("evaluation_failures", [])
                total_failures = len(failures) + len(eval_failures)

                if total_failures > 0:
                    print(f"\n⚠️  Failures: {total_failures}")
                    if failures:
                        print(f"  Generation failures: {len(failures)}")
                        if verbose:
                            for failure in failures[:3]:
                                print(
                                    f"    - {failure.get('sample_id')}: {failure.get('message')}"
                                )
                    if eval_failures:
                        print(f"  Evaluation failures: {len(eval_failures)}")

                # Show sample outputs
                if show_outputs:
                    samples = report_data.get("samples", [])
                    print("\n📝 Sample Outputs (showing up to 3):")
                    for i, sample in enumerate(samples[:3], 1):
                        sample_id = sample.get("sample_id", f"sample-{i}")
                        scores = sample.get("scores", [])

                        print(f"\n  Sample {i}: {sample_id}")
                        if scores:
                            for score in scores:
                                metric_name = score.get("metric")
                                value = score.get("value")
                                print(f"    {metric_name}: {value:.4f}")

                # Summary
                print("\n" + "=" * 80)
                print("✨ Next Steps:")
                print("=" * 80)

                if total_failures == 0 and metrics:
                    avg_metric = metrics[0]["mean"]
                    if avg_metric > 0.1:  # Reasonable performance
                        print("  ✅ Configuration looks good!")
                        print("  Run full experiment with:")
                        print(
                            f"     uv run python -m themis.cli run-config --config {config_path}"
                        )
                    else:
                        print("  ⚠️  Low performance on sample - consider:")
                        print("     - Adjusting prompt template")
                        print("     - Tuning temperature/max_tokens")
                        print("     - Testing different model")
                else:
                    print("  ⚠️  Issues detected:")
                    if total_failures > 0:
                        print("     - Fix failures before full run")
                    if not metrics:
                        print("     - Check evaluation metrics")
                    print("     - Review configuration")

                return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        if verbose:
            traceback.print_exc()
        return 1


__all__ = ["sample_run_command"]
