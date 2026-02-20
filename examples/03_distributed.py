"""Execution tuning example (workers + retries) with evaluate().

This replaces the old distributed placeholder.
"""

import themis
from themis.presets import get_benchmark_preset

preset = get_benchmark_preset("demo")

report = themis.evaluate(
    preset.load_dataset(limit=10),
    model="fake:fake-math-llm",
    prompt=preset.prompt_template.template,
    metrics=[m.name for m in preset.metrics],
    temperature=0.0,
    max_tokens=128,
    workers=4,
    max_retries=2,
    storage=".cache/experiments",
    resume=True,
)

print("Completed", len(report.generation_results), "samples")
print("Failures:", len(report.failures))
