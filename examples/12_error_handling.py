"""Example: Error handling patterns in Themis.

This script demonstrates how to catch and handle specific exceptions
that may arise during configuration, generation, or evaluation.
"""

import logging

import themis
from themis.exceptions import ProviderError, ConfigurationError, MetricError

# Setup basic logging to see Themis's internal warnings
logging.basicConfig(level=logging.INFO)


def main() -> None:
    # 1. Handling Provider Errors
    # If a model or provider is incorrectly specified, or an API key is missing
    print("--- 1. Provider Error Handling ---")
    try:
        themis.evaluate(
            "demo",
            model="unknown_provider:gpt-4",
            limit=2,
        )
    except ProviderError as e:
        print(f"Caught ProviderError: {e}")

    # 2. Handling Configuration Errors
    # If a preset or benchmark configuration is unknown
    print("\n--- 2. Configuration Error Handling ---")
    try:
        themis.evaluate(
            "nonexistent_benchmark",
            model="fake:fake-math-llm",
            limit=2,
        )
    except ConfigurationError as e:
        print(f"Caught ConfigurationError: {e}")

    # 3. Handling Metric Errors
    # If a custom metric is requested but not registered
    print("\n--- 3. Metric Error Handling ---")
    try:
        themis.evaluate(
            "demo",
            model="fake:fake-math-llm",
            metrics=["unknown_metric_123"],
            limit=2,
        )
    except MetricError as e:
        print(f"Caught MetricError: {e}")


if __name__ == "__main__":
    main()
