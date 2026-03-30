"""Dataset loaders for quick-eval workflows."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from themis.core.models import Case, Dataset


class MissingOptionalDependencyError(RuntimeError):
    """Raised when an optional integration dependency is unavailable."""


def dataset_from_inline(
    *,
    input_value,
    expected_output=None,
    dataset_id: str = "inline",
    case_id: str = "case-1",
    revision: str = "inline",
) -> Dataset:
    return Dataset(
        dataset_id=dataset_id,
        revision=revision,
        cases=[Case(case_id=case_id, input=input_value, expected_output=expected_output)],
    )


def dataset_from_jsonl(path: str | Path, *, dataset_id: str | None = None, revision: str | None = None) -> Dataset:
    source_path = Path(path)
    cases: list[Case] = []
    for index, line in enumerate(source_path.read_text().splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            Case(
                case_id=str(payload.get("case_id", f"case-{index + 1}")),
                input=payload["input"],
                expected_output=payload.get("expected_output"),
                metadata={key: str(value) for key, value in payload.get("metadata", {}).items()},
            )
        )
    return Dataset(
        dataset_id=dataset_id or source_path.stem,
        revision=revision,
        cases=cases,
    )


def dataset_from_huggingface(
    *,
    dataset_name: str,
    split: str,
    input_field: str,
    expected_output_field: str | None = None,
    case_id_field: str | None = None,
) -> Dataset:
    try:
        datasets_module = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError(
            "Hugging Face quick-eval requires the optional datasets dependency. "
            "Install it with: pip install themis-eval[datasets]"
        ) from exc

    rows = datasets_module.load_dataset(dataset_name, split=split)
    cases: list[Case] = []
    for index, row in enumerate(rows):
        if input_field not in row:
            raise ValueError(f"Missing input field '{input_field}' in dataset row {index}")
        case_id = str(row.get(case_id_field, f"{split}-{index}")) if case_id_field is not None else f"{split}-{index}"
        expected_output = row.get(expected_output_field) if expected_output_field is not None else None
        metadata = {
            key: str(value)
            for key, value in row.items()
            if key not in {input_field, expected_output_field, case_id_field}
        }
        cases.append(
            Case(
                case_id=case_id,
                input=row[input_field],
                expected_output=expected_output,
                metadata=metadata,
            )
        )

    return Dataset(dataset_id=dataset_name, revision=split, cases=cases)
