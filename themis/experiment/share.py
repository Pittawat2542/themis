"""Shareable assets for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any

from themis.exceptions import MetricError
from themis.storage import ExperimentStorage


@dataclass(frozen=True)
class ShareSummary:
    """Normalized summary details for sharing."""

    run_id: str
    metrics: dict[str, dict[str, float | int | None]]
    total_samples: int | None
    model: str | None
    cost_usd: float | None


@dataclass(frozen=True)
class SharePack:
    """Generated share assets."""

    svg_path: Path
    markdown_path: Path
    markdown_snippet: str
    event_log_path: Path | None


def create_share_pack(
    *,
    run_id: str,
    storage_root: Path,
    output_dir: Path,
    metric: str | None = None,
) -> SharePack:
    """Generate a shareable SVG badge + Markdown snippet for a run."""
    summary = _load_share_summary(run_id=run_id, storage_root=storage_root)
    metric_name, metric_value = _select_metric(summary.metrics, metric)

    svg = _render_share_svg(
        run_id=summary.run_id,
        metric_name=metric_name,
        metric_value=metric_value,
        model=summary.model,
        total_samples=summary.total_samples,
        cost_usd=summary.cost_usd,
    )

    safe_run_id = _sanitize_filename(summary.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"themis-share-{safe_run_id}.svg"
    svg_path.write_text(svg, encoding="utf-8")

    markdown_snippet = f"![Themis result]({_relative_markdown_path(svg_path)})"
    markdown_path = output_dir / f"themis-share-{safe_run_id}.md"
    markdown_path.write_text(markdown_snippet + "\n", encoding="utf-8")

    event_log_path = _log_share_event(
        storage_root=storage_root,
        event_name="share_pack_generated",
        payload={
            "run_id": summary.run_id,
            "metric": metric_name,
            "metric_value": metric_value,
            "output_dir": str(output_dir),
            "svg_path": str(svg_path),
        },
    )

    return SharePack(
        svg_path=svg_path,
        markdown_path=markdown_path,
        markdown_snippet=markdown_snippet,
        event_log_path=event_log_path,
    )


def _load_share_summary(*, run_id: str, storage_root: Path) -> ShareSummary:
    storage = ExperimentStorage(storage_root)
    run_path = storage.get_run_path(run_id)
    summary_path = run_path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {}) or {}
        return ShareSummary(
            run_id=str(summary.get("run_id") or run_id),
            metrics=metrics,
            total_samples=_safe_int(summary.get("total_samples")),
            model=_safe_str(summary.get("metadata", {}).get("model")),
            cost_usd=_safe_float(summary.get("cost_usd")),
        )

    report_path = run_path / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = {
            entry.get("name"): {
                "mean": entry.get("mean"),
                "count": entry.get("count"),
            }
            for entry in report.get("metrics", [])
            if entry.get("name")
        }
        summary = report.get("summary", {}) or {}
        model = None
        samples = report.get("samples", [])
        if samples:
            metadata = samples[0].get("metadata", {}) or {}
            model = _safe_str(metadata.get("model_identifier") or metadata.get("model"))
        cost_usd = None
        cost = summary.get("cost")
        if isinstance(cost, dict):
            cost_usd = _safe_float(cost.get("total_cost"))
        total_samples = _safe_int(summary.get("total_samples"))
        if total_samples is None:
            total_samples = _safe_int(report.get("total_samples"))
        return ShareSummary(
            run_id=str(summary.get("run_id") or run_id),
            metrics=metrics,
            total_samples=total_samples,
            model=model,
            cost_usd=cost_usd,
        )

    raise FileNotFoundError(
        f"Run {run_id} is missing summary.json or report.json at {run_path}"
    )


def _select_metric(
    metrics: dict[str, dict[str, float | int | None]],
    metric: str | None,
) -> tuple[str, float | None]:
    if not metrics:
        raise MetricError("No metrics found for this run")

    if metric is None:
        metric = sorted(metrics.keys())[0]

    if metric not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise MetricError(f"Metric '{metric}' not found. Available: {available}")

    mean = metrics[metric].get("mean")
    return metric, _safe_float(mean)


def _render_share_svg(
    *,
    run_id: str,
    metric_name: str,
    metric_value: float | None,
    model: str | None,
    total_samples: int | None,
    cost_usd: float | None,
) -> str:
    title = "Themis Result"
    metric_display = (
        f"{metric_name}: {metric_value:.4f}" if metric_value is not None else "N/A"
    )
    meta_lines = []
    meta_lines.append(f"Model: {model or 'unknown'}")
    meta_lines.append(
        f"Samples: {total_samples if total_samples is not None else 'N/A'}"
    )
    if cost_usd is not None:
        meta_lines.append(f"Cost: ${cost_usd:.4f}")

    line_height = 18
    base_y = 64
    run_line_y = base_y + (len(meta_lines) * line_height) + 10
    height = run_line_y + 24

    meta_svg = "\n".join(
        f'<text class="meta" x="24" y="{base_y + (idx * line_height)}">'
        f"{html.escape(line)}</text>"
        for idx, line in enumerate(meta_lines)
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" viewBox="0 0 640 {height}">
  <defs>
    <style>
      .title {{ font: 600 18px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; fill: #0f172a; }}
      .metric {{ font: 700 22px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; fill: #0f172a; }}
      .meta {{ font: 400 14px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; fill: #334155; }}
      .run {{ font: 400 12px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; fill: #64748b; }}
    </style>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#e2e8f0"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="640" height="{height}" rx="16" fill="url(#bg)"/>
  <rect x="16" y="14" width="608" height="{height - 28}" rx="12" fill="#ffffff" stroke="#e2e8f0"/>
  <text class="title" x="24" y="38">{html.escape(title)}</text>
  <text class="metric" x="24" y="60">{html.escape(metric_display)}</text>
  {meta_svg}
  <text class="run" x="24" y="{run_line_y}">Run: {html.escape(run_id)}</text>
</svg>"""


def _log_share_event(
    *,
    storage_root: Path,
    event_name: str,
    payload: dict[str, Any],
) -> Path | None:
    try:
        events_path = storage_root / "share_events.jsonl"
        event = {
            "event": event_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        return events_path
    except OSError:
        return None


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _relative_markdown_path(path: Path) -> str:
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        relative = path
    return relative.as_posix()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
