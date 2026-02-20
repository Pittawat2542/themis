"""Provider-ready evaluation example.

Runs with a real provider if `THEMIS_EXAMPLE_MODEL` is set, otherwise falls
back to fake model for local smoke testing.
"""

from __future__ import annotations

import os
from datetime import datetime

from themis.api import evaluate

model = os.getenv("THEMIS_EXAMPLE_MODEL", "fake-math-llm")
run_id = f"provider-ready-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

report = evaluate(
    "demo",
    model=model,
    limit=5,
    temperature=0.0,
    run_id=run_id,
)

print("Model:", model)
print("Run ID:", report.metadata.get("run_id"))
for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
