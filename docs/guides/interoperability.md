# Interoperability Guide

This guide explains how to export Themis results and load them into common tools.

## Export formats

Themis exports are created via:
- `themis.experiment.export.export_report_json`
- `themis.experiment.export.export_report_csv`
- `themis.experiment.export.export_html_report`

### JSON report schema (export_report_json)

Top-level keys:
- `title`: Report title string.
- `summary`: Run metadata and counts (includes `run_failures` and `evaluation_failures`).
- `metrics`: List of aggregate metrics with `name`, `count`, and `mean`.
- `samples`: Per-sample results:
  - `sample_id`: Dataset id (string).
  - `metadata`: Flattened metadata from generation records.
  - `scores`: List of `{metric, value, details, metadata}` objects.
  - `failures`: List of evaluation failure messages.
- `rendered_sample_limit`: Sample limit used for the report.
- `total_samples`: Total evaluation records.
- `charts`: Optional chart data if provided.
- `run_failures`: Generation failures (sample id + message).
- `evaluation_failures`: Evaluation failures (sample id + message).
- `metrics_rendered`: List of metric names rendered.

Minimal example:
```json
{
  "title": "Experiment report",
  "summary": {
    "run_id": "gsm8k-gpt4-2024-01-01",
    "total_samples": 100,
    "run_failures": 0,
    "evaluation_failures": 0
  },
  "metrics": [
    {"name": "exact_match", "count": 100, "mean": 0.82}
  ],
  "samples": [
    {
      "sample_id": "gsm8k-00001",
      "metadata": {"prompt_template": "gsm8k-zero-shot"},
      "scores": [{"metric": "exact_match", "value": 1.0, "details": {}, "metadata": {}}],
      "failures": []
    }
  ]
}
```

### CSV schema (export_report_csv)

CSV columns:
- `sample_id`
- One column per metadata field found in the run.
- One column per metric: `metric:{metric_name}`
- `failures` (optional)

Example header:
```
sample_id,subject,metric:exact_match,failures
```

### HTML report (export_html_report)

The HTML report is a rendered summary designed for human review (no strict schema).
It includes aggregated metrics, sample rows, and optional charts.

## Mappings to common tools

### Hugging Face Datasets

Use the JSON report to build a Dataset of per-sample metrics.
Requires `datasets`:
```bash
uv add datasets
```

```python
import json
from datasets import Dataset

with open("report.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

rows = []
for sample in payload["samples"]:
    row = {
        "sample_id": sample["sample_id"],
        **sample.get("metadata", {}),
    }
    for score in sample.get("scores", []):
        row[f"metric:{score['metric']}"] = score["value"]
    rows.append(row)

dataset = Dataset.from_list(rows)
```

If you want to upload to the Hub, see `themis/integrations/huggingface.py`.

### Weights & Biases (W&B)

Requires `wandb`:
```bash
uv add wandb
```

```python
import json
import wandb

wandb.init(project="themis-eval")

with open("report.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

summary = {m["name"]: m["mean"] for m in payload.get("metrics", [])}
wandb.summary.update(summary)

table = wandb.Table(columns=["sample_id", "metric", "value"])
for sample in payload.get("samples", []):
    for score in sample.get("scores", []):
        table.add_data(sample["sample_id"], score["metric"], score["value"])

wandb.log({"samples": table})
```

For a built-in integration, see `themis/integrations/wandb.py`.

### MLflow

Requires `mlflow`:
```bash
uv add mlflow
```

```python
import json
import mlflow

with open("report.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

with mlflow.start_run(run_name=payload.get("summary", {}).get("run_id")):
    for metric in payload.get("metrics", []):
        mlflow.log_metric(metric["name"], metric["mean"])

    mlflow.log_artifact("report.json")
    mlflow.log_artifact("report.csv")
    mlflow.log_artifact("report.html")
```

## CLI export reminder

The CLI supports exports via `--output`:
```bash
themis eval gsm8k --model gpt-4 --limit 100 --output results.json
themis eval gsm8k --model gpt-4 --limit 100 --output results.csv
themis eval gsm8k --model gpt-4 --limit 100 --output results.html
```

## Related Tutorial

- Countdown Part 5 covers reproducibility gates and bundle indexing.
- Countdown Part 6 covers backend wiring plus W&B/MLflow publication from the same bundle.
- Countdown Part 7 covers storage observability/retention and optional Hugging Face Hub publication.
- Countdown Part 8 covers statistical robustness and reliability analysis.
- Countdown Part 9 covers manifest audits and reproducibility-diff gates.
- Full guide: [Countdown Tutorial](countdown-tutorial.md)
