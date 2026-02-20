"""Quick start with themis.evaluate()."""

import themis
from themis.presets import get_benchmark_preset

preset = get_benchmark_preset("demo")

report = themis.evaluate(
    preset.load_dataset(limit=5),
    model="fake:fake-math-llm",
    prompt=preset.prompt_template.template,
    metrics=[m.name for m in preset.metrics],
    temperature=0.0,
    max_tokens=128,
    workers=2,
    storage=".cache/experiments",
    resume=True,
)

print("Run:", report.metadata.get("run_id"))
for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
