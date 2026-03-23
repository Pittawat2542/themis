"""Dataset provider for m-a-p/AetherCode."""

from __future__ import annotations

from themis._optional import import_optional
from themis.specs.foundational import DatasetSpec

from ...datasets._loaders import (
    _is_dataset_generation_error,
    _prepare_huggingface_dataset_for_iteration,
)
from ...datasets._normalizers import _metadata_dict
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows

DEFAULT_AETHERCODE_SUBSET = "v1_2024"


class BuiltinAetherCodeDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or _load_aethercode_rows)

    def load_rows(self, dataset: DatasetSpec) -> list[dict[str, object]]:
        if self._huggingface_loader is _load_aethercode_rows:
            if dataset.dataset_id is None:
                raise ValueError(
                    "Built-in HuggingFace dataset providers require a dataset_id."
                )
            return _load_aethercode_rows(
                dataset.dataset_id,
                dataset.split,
                dataset.revision,
            )
        return super().load_rows(dataset)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        del dataset
        normalized: list[dict[str, object]] = []
        skipped_missing_tests = 0
        for row in rows:
            payload = dict(row)
            tests = _normalize_aethercode_tests(payload.get("test_cases"))
            if not tests:
                skipped_missing_tests += 1
                continue
            payload["item_id"] = str(payload.get("id", payload.get("item_id", "")))
            payload["prompt_text"] = _aethercode_prompt(payload)
            payload["language"] = "cpp"
            payload["execution_mode"] = "stdio"
            payload["input_mode"] = "stdio"
            payload["official_tests"] = tests
            payload["time_limit"] = _milliseconds_to_seconds(payload.get("time_limit"))
            checker = _normalize_checker(payload.get("checker"))
            payload["generated_checker"] = checker
            if checker is not None:
                payload["checker_language"] = "cpp"
            payload["metadata"] = _metadata_dict(
                payload,
                [
                    "difficulty",
                    "contest_category",
                    "contest_name",
                    "date",
                    "year",
                ],
            )
            normalized.append(payload)
        return CatalogNormalizedRows(
            rows=normalized,
            stats={"skipped_missing_tests_count": skipped_missing_tests},
        )


def _load_aethercode_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    *,
    datasets_module=None,
) -> list[dict[str, object]]:
    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        dataset = datasets.load_dataset(
            dataset_id,
            DEFAULT_AETHERCODE_SUBSET,
            split=split,
            revision=revision,
        )
    except Exception as exc:
        if not _is_dataset_generation_error(exc, datasets):
            raise
        dataset = datasets.load_dataset(
            dataset_id,
            DEFAULT_AETHERCODE_SUBSET,
            split=split,
            revision=revision,
            streaming=True,
        )
    dataset = _prepare_huggingface_dataset_for_iteration(dataset, datasets)
    return [dict(row) for row in dataset]


def _normalize_aethercode_tests(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    tests: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        raw_input = entry.get("input")
        raw_output = entry.get("output")
        if isinstance(raw_input, str) and isinstance(raw_output, str):
            tests.append({"input": raw_input, "output": raw_output})
    return tests


def _normalize_checker(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _milliseconds_to_seconds(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) / 1000.0
    if isinstance(value, str) and value.strip():
        try:
            return float(value) / 1000.0
        except ValueError:
            return None
    return None


def _aethercode_prompt(payload: dict[str, object]) -> str:
    description = str(payload.get("description", "")).strip()
    return (
        "Write a C++17 program that solves the following problem. "
        "Return only code.\n\n"
        f"Problem:\n{description}"
    )
