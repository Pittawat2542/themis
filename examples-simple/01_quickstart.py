"""Quick start example - Run your first evaluation in 10 lines.

This example shows how to run a benchmark evaluation with Themis
using the new unified API. No boilerplate, just results.
"""

import themis

# Run evaluation with one line
report = themis.evaluate(
    "demo",  # Use the demo benchmark for testing
    model="fake-math-llm",  # Use fake model for testing (no API key needed)
    limit=3,  # Evaluate only 3 samples
)

# Print results
print("\nEvaluation Results:")
print(f"Samples evaluated: {len(report.generation_results)}")

if report.evaluation_report.aggregates:
    for aggregate in report.evaluation_report.aggregates:
        print(f"{aggregate.metric_name}: {aggregate.mean:.2%}")
