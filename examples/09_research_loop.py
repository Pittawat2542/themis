"""End-to-end research loop: run -> export -> compare -> share."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from themis.api import evaluate
from themis.experiment.comparison import compare_runs
from themis.experiment.comparison import StatisticalTest
from themis.experiment import export
from themis.experiment.share import create_share_pack

storage_root = Path(".cache/experiments")
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_a = f"research-a-{stamp}"
run_b = f"research-b-{stamp}"

report_a = evaluate(
    "demo",
    model="fake-math-llm",
    limit=10,
    temperature=0.0,
    run_id=run_a,
)

report_b = evaluate(
    "demo",
    model="fake-math-llm",
    limit=10,
    temperature=0.7,
    run_id=run_b,
)

output_dir = Path(".cache/example_outputs") / f"research-loop-{stamp}"
output_dir.mkdir(parents=True, exist_ok=True)

export.export_report_json(report_a, output_dir / "run-a-report.json")
export.export_report_csv(report_a, output_dir / "run-a-report.csv")
export.export_html_report(report_a, output_dir / "run-a-report.html")

comparison = compare_runs(
    run_ids=[run_a, run_b],
    storage_path=storage_root,
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,
)

share = create_share_pack(
    run_id=run_a,
    storage_root=storage_root,
    output_dir=output_dir,
)

print("Run A:", run_a)
print("Run B:", run_b)
print("Outputs:", output_dir)
print("Comparison best:", comparison.overall_best_run)
print("Share markdown:", share.markdown_path)
