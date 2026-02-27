"""CLI utility functions."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

from themis.exceptions import DatasetError

from themis.experiment import export as experiment_export
from themis.experiment import orchestrator


def export_outputs(
    report: orchestrator.ExperimentReport,
    *,
    csv_output: Path | None,
    html_output: Path | None,
    json_output: Path | None,
    title: str,
) -> None:
    """Export experiment report to various formats.

    Args:
        report: Experiment report to export
        csv_output: Optional path for CSV export
        html_output: Optional path for HTML export
        json_output: Optional path for JSON export
        title: Title for the report
    """
    outputs = experiment_export.export_report_bundle(
        report,
        csv_path=csv_output,
        html_path=html_output,
        json_path=json_output,
        title=title,
    )
    for kind, output_path in outputs.items():
        print(f"Exported {kind.upper()} to {output_path}")


def effective_total(total: int, limit: int | None) -> int:
    """Calculate effective total based on limit.

    Args:
        total: Total number of items
        limit: Optional limit

    Returns:
        Effective total (min of total and limit)
    """
    if limit is None:
        return total
    return min(total, limit)


# --- Storage Helpers ---


def resolve_storage_root(storage: str | None) -> Path:
    """Resolve storage root path."""
    import os

    if storage:
        return Path(storage).expanduser()
    env_storage = os.getenv("THEMIS_STORAGE")
    if env_storage:
        return Path(env_storage).expanduser()
    return Path(".cache/experiments")


# --- Dataset Helpers ---


_PROMPT_FIELD_CANDIDATES = ("prompt", "question", "input", "text", "query", "problem")
_REFERENCE_FIELD_CANDIDATES = (
    "answer",
    "reference",
    "expected",
    "label",
    "target",
    "solution",
)
_ID_FIELD_CANDIDATES = ("id", "sample_id", "dataset_id", "unique_id", "uid")


def load_custom_dataset_file(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Load and normalize a custom dataset file."""
    rows = _read_dataset_rows(path)
    if not rows:
        raise DatasetError(f"Dataset file is empty: {path}")

    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise DatasetError(
                f"Row {index} in {path} must be a JSON object, got {type(row).__name__}."
            )
        normalized_rows.append(dict(row))

    prompt_field = _detect_required_field(
        normalized_rows, _PROMPT_FIELD_CANDIDATES, "prompt"
    )
    reference_field = _detect_required_field(
        normalized_rows, _REFERENCE_FIELD_CANDIDATES, "reference"
    )
    id_field = _detect_optional_field(normalized_rows, _ID_FIELD_CANDIDATES)

    for index, row in enumerate(normalized_rows, 1):
        if id_field is None:
            row["id"] = str(index)
        elif id_field != "id":
            row["id"] = row[id_field]

        if reference_field not in ("answer", "reference"):
            row.setdefault("reference", row[reference_field])

    return normalized_rows, prompt_field, reference_field


def _read_dataset_rows(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                content = line.strip()
                if not content:
                    continue
                try:
                    rows.append(json.loads(content))
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"Invalid JSON on line {line_no} in {path}: {exc.msg}"
                    ) from exc
        return rows
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON in {path}: {exc.msg}") from exc
        if not isinstance(payload, builtins.list):
            raise DatasetError(
                f"JSON dataset in {path} must be a top-level array of row objects."
            )
        return payload
    raise DatasetError(
        f"Unsupported dataset format '{suffix or '<none>'}' for {path}. "
        "Use .json or .jsonl."
    )


def _detect_required_field(
    rows: list[dict[str, Any]], candidates: tuple[str, ...], kind: str
) -> str:
    field = _detect_optional_field(rows, candidates)
    if field is None:
        options = ", ".join(candidates)
        raise DatasetError(
            f"Could not detect {kind} field in dataset. Add one of: {options}."
        )
    return field


def _detect_optional_field(
    rows: list[dict[str, Any]], candidates: tuple[str, ...]
) -> str | None:
    for candidate in candidates:
        if all(candidate in row for row in rows):
            return candidate
    return None
