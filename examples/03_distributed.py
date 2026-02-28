"""Execution resilience and parallelization example with `themis.evaluate()`.

This example demonstrates how to scale evaluations and handle rate limits natively.
"""

import themis
from themis.presets import get_benchmark_preset

# 1. Load the benchmark preset
preset = get_benchmark_preset("demo")

# 2. Run the evaluation with parallel workers and retry logic
report = themis.evaluate(
    preset.load_dataset(limit=10),
    model="fake:fake-math-llm",
    prompt=preset.prompt_template.template,
    metrics=[m.name for m in preset.metrics],
    temperature=0.0,
    max_tokens=128,
    # --- Execution Tuning ---
    workers=4,  # Run 4 requests concurrently
    max_retries=2,  # Retry twice if the API returns an error (e.g., HTTP 429)
    # timeout=30,        # You can pass LiteLLM kwargs natively
    storage=".cache/experiments",
    resume=True,
)

print("Completed", len(report.generation_results), "samples")
print("Failures:", len(report.failures))  # Inspect failures that exceeded max_retries
