"""Dataset inspection helpers for the catalog."""

from __future__ import annotations

import json

from themis._optional import import_optional
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict

from ._loaders import _invoke_huggingface_loader, load_huggingface_rows
from ._types import CatalogMetadataLoader, CatalogRow, CatalogRowLoader


def inspect_huggingface_dataset(
    dataset_id: str,
    *,
    config_name: str | None = None,
    split: str = "test",
    revision: str | None = None,
    metadata_loader: CatalogMetadataLoader | None = None,
    row_loader: CatalogRowLoader | None = None,
    max_samples: int = 3,
    datasets_module=None,
) -> JSONDict:
    datasets = datasets_module or import_optional("datasets", extra="datasets")
    metadata = (
        metadata_loader(dataset_id, revision)
        if metadata_loader is not None
        else _default_dataset_metadata(
            dataset_id,
            revision,
            datasets_module=datasets,
        )
    )
    rows = (
        _invoke_huggingface_loader(
            row_loader,
            dataset_id,
            split,
            revision,
            config_name=config_name,
        )
        if row_loader is not None
        else load_huggingface_rows(
            dataset_id,
            split,
            revision,
            config_name,
            datasets_module=datasets,
        )
    )
    return validate_json_dict(
        {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
            "revision": revision,
            "splits": _string_list(metadata.get("splits")),
            "gated": bool(metadata.get("gated", False)),
            "modalities": _string_list(metadata.get("modalities")),
            "fields": _infer_field_types(rows),
            "row_count": len(rows),
            "samples": _json_rows(rows[:max_samples], label="catalog dataset samples"),
            **_suggest_dataset_wiring(rows),
        },
        label="catalog dataset inspection",
    )


def _default_dataset_metadata(
    dataset_id: str,
    revision: str | None,
    *,
    datasets_module=None,
) -> JSONDict:
    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        builder = datasets.load_dataset_builder(dataset_id, revision=revision)
    except Exception as exc:
        message = str(exc).lower()
        return validate_json_dict(
            {
                "dataset_id": dataset_id,
                "gated": "agree to share your contact information" in message
                or "access" in message
                and "denied" in message,
                "splits": [],
                "modalities": [],
                "error": str(exc),
            },
            label="catalog dataset metadata",
        )
    features = getattr(builder.info, "features", {}) or {}
    splits = sorted(list((getattr(builder.info, "splits", {}) or {}).keys()))
    modalities = _infer_modalities_from_features(features)
    return validate_json_dict(
        {
            "dataset_id": dataset_id,
            "gated": False,
            "splits": splits,
            "modalities": modalities,
        },
        label="catalog dataset metadata",
    )


def _infer_modalities_from_features(features: object) -> list[str]:
    text_like = False
    image_like = False
    serialized = json.dumps(str(features)).lower()
    if "image" in serialized:
        image_like = True
    if "string" in serialized or "value('string')" in serialized:
        text_like = True
    modalities: list[str] = []
    if text_like:
        modalities.append("text")
    if image_like:
        modalities.append("image")
    return modalities


def _infer_field_types(rows: list[CatalogRow]) -> JSONDict:
    field_types: JSONDict = {}
    for row in rows:
        for key, value in row.items():
            if key in field_types or value is None:
                continue
            field_types[key] = type(value).__name__
    return validate_json_dict(field_types, label="catalog dataset fields")


def _suggest_dataset_wiring(rows: list[CatalogRow]) -> JSONDict:
    if not rows:
        return validate_json_dict(
            {
                "suggested_prompt_field": None,
                "suggested_answer_field": None,
                "suggested_item_id_field": None,
                "suggested_metadata_keys": [],
            },
            label="catalog dataset suggestions",
        )
    row = rows[0]
    field_names = {str(key) for key in row}
    prompt_field = _first_present_field(
        field_names,
        ["problem", "question", "prompt", "Problem"],
    )
    answer_field = _first_present_field(
        field_names,
        ["answer", "expected", "Short Answer", "answer_letter"],
    )
    item_id_field = _first_present_field(
        field_names,
        ["problem_idx", "Problem ID", "item_id", "id", "question_id", "prompt_id"],
    )
    ignored = {
        prompt_field,
        answer_field,
        item_id_field,
        "item_id",
        "__index_level_0__",
    }
    metadata_keys: list[str] = []
    for key, value in row.items():
        normalized_key = str(key)
        if normalized_key in ignored or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata_keys.append(normalized_key)
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            metadata_keys.append(normalized_key)
    return validate_json_dict(
        {
            "suggested_prompt_field": prompt_field,
            "suggested_answer_field": answer_field,
            "suggested_item_id_field": item_id_field,
            "suggested_metadata_keys": metadata_keys,
        },
        label="catalog dataset suggestions",
    )


def _first_present_field(field_names: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    return None


def _json_rows(rows: list[CatalogRow], *, label: str) -> list[JSONDict]:
    return [
        validate_json_dict(dict(row), label=f"{label}[{index}]")
        for index, row in enumerate(rows)
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
