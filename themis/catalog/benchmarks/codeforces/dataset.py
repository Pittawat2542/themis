"""Dataset provider for open-r1/codeforces verifiable prompts."""

from __future__ import annotations

from themis._optional import import_optional
from themis.specs.foundational import DatasetSpec

from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _metadata_dict,
    _prepare_huggingface_dataset_for_iteration,
)


class BuiltinOpenR1CodeforcesDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or _load_codeforces_rows)

    def load_rows(self, dataset: DatasetSpec) -> list[dict[str, object]]:
        if self._huggingface_loader is _load_codeforces_rows:
            if dataset.dataset_id is None:
                raise ValueError(
                    "Built-in HuggingFace dataset providers require a dataset_id."
                )
            return _load_codeforces_rows(
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
        skipped_file_mode = 0
        skipped_interactive = 0
        skipped_incomplete_tests = 0
        for row in rows:
            payload = dict(row)
            if str(payload.get("input_mode", "")).strip().lower() != "stdio":
                skipped_file_mode += 1
                continue
            interaction_format = payload.get("interaction_format")
            if isinstance(interaction_format, str) and interaction_format.strip():
                skipped_interactive += 1
                continue
            if not bool(payload.get("official_tests_complete", False)):
                skipped_incomplete_tests += 1
                continue
            payload["item_id"] = str(payload.get("id", payload.get("item_id", "")))
            _validate_required_codeforces_fields(payload)
            payload["metadata"] = _metadata_dict(
                payload,
                ["contest_id", "language", "rating", "input_mode"],
            )
            normalized.append(payload)
        return CatalogNormalizedRows(
            rows=normalized,
            stats={
                "skipped_file_mode_count": skipped_file_mode,
                "skipped_interactive_count": skipped_interactive,
                "skipped_incomplete_tests_count": skipped_incomplete_tests,
            },
        )


def _load_codeforces_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    *,
    datasets_module=None,
) -> list[dict[str, object]]:
    datasets = datasets_module or import_optional("datasets", extra="datasets")
    dataset = datasets.load_dataset(
        dataset_id,
        "verifiable-prompts",
        split=split,
        revision=revision,
    )
    dataset = _prepare_huggingface_dataset_for_iteration(dataset, datasets)
    return [dict(row) for row in dataset]


def _validate_required_codeforces_fields(payload: dict[str, object]) -> None:
    missing_fields = [
        field_name
        for field_name in ("prompt", "language")
        if not isinstance(payload.get(field_name), str)
        or not str(payload.get(field_name)).strip()
    ]
    if not missing_fields:
        return
    item_id = str(payload.get("id", payload.get("item_id", "<unknown>")))
    joined = ", ".join(missing_fields)
    raise ValueError(
        f"codeforces benchmark row '{item_id}' is missing required field(s): {joined}."
    )
