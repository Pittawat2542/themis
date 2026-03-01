"""Quick start with themis.evaluate()."""

import themis

report = themis.evaluate(
    "demo",
    model="fake:fake-math-llm",
    limit=5,
    temperature=0.0,
    max_tokens=128,
    workers=2,
    storage=".cache/experiments",
)

print("Run:", report.metadata.get("run_id"))
for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
