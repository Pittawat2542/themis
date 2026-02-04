"""Resume + cache behavior example."""

from __future__ import annotations

import time
from datetime import datetime

from themis.api import evaluate

run_id = f"resume-cache-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

start = time.perf_counter()
report_first = evaluate(
    "demo",
    model="fake-math-llm",
    limit=10,
    run_id=run_id,
    resume=False,
)
first_elapsed = time.perf_counter() - start

start = time.perf_counter()
report_resume = evaluate(
    "demo",
    model="fake-math-llm",
    limit=10,
    run_id=run_id,
    resume=True,
)
resume_elapsed = time.perf_counter() - start

print("Run ID:", run_id)
print(f"First run:  {first_elapsed:.3f}s ({len(report_first.generation_results)} samples)")
print(f"Resume run: {resume_elapsed:.3f}s ({len(report_resume.generation_results)} samples)")
