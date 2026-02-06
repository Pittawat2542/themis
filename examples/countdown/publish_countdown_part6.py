from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


if __name__ == "__main__":
    bundle_dir = Path("outputs/countdown_part6")
    report_json = bundle_dir / "report.json"
    report_csv = bundle_dir / "report.csv"
    report_html = bundle_dir / "report.html"

    if not report_json.exists():
        raise SystemExit("Missing outputs/countdown_part6/report.json. Run run_countdown_part6_backends.py first.")

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    run_name = payload.get("summary", {}).get("run_id", "countdown-part6")
    metrics = payload.get("metrics", [])
    samples = payload.get("samples", [])

    if importlib.util.find_spec("wandb"):
        os.environ.setdefault("WANDB_MODE", "offline")
        import wandb

        wandb.init(project="themis-countdown", name=run_name)
        for metric in metrics:
            wandb.summary[f"{metric['name']}_mean"] = metric["mean"]

        table = wandb.Table(columns=["sample_id", "metric", "value"])
        for sample in samples:
            for score in sample.get("scores", []):
                table.add_data(sample["sample_id"], score["metric"], score["value"])
        wandb.log({"countdown_samples": table})
        wandb.finish()
        print("wandb", "published")
    else:
        print("wandb", "skipped (not installed)")

    if importlib.util.find_spec("mlflow"):
        import mlflow

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("themis-countdown")
        with mlflow.start_run(run_name=run_name):
            for metric in metrics:
                mlflow.log_metric(metric["name"], metric["mean"])
            mlflow.log_artifact(str(report_json))
            mlflow.log_artifact(str(report_csv))
            mlflow.log_artifact(str(report_html))
        print("mlflow", "published")
    else:
        print("mlflow", "skipped (not installed)")
