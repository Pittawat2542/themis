"""Dataset provider for ZenMoore/RoleBench."""

from __future__ import annotations

from urllib import request

from ..._http import DEFAULT_HTTP_TIMEOUT_SECONDS, iter_jsonl_url, load_json_url
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows

_ROLEBENCH_VARIANT_PATHS = {
    "instruction_generalization_eng": "instruction-generalization",
    "role_generalization_eng": "role-generalization",
}
_ROLEBENCH_SUBSETS = ("general", "role_specific")


class BuiltinRoleBenchDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or _load_rolebench_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        variant_id = _resolve_rolebench_variant_id(dataset)
        normalized: list[dict[str, object]] = []
        for row in rows:
            payload = dict(row)
            subset = str(payload.get("subset", "general"))
            source_line_value = payload.get("source_line_number", 1)
            source_line_number = (
                source_line_value
                if isinstance(source_line_value, int)
                else int(str(source_line_value))
            )
            item_id = payload.get("item_id")
            if (
                not isinstance(item_id, str)
                or not item_id
                or item_id.startswith("item-")
            ):
                item_id = f"rolebench-{variant_id}-{subset}-{source_line_number}"
            generated = payload.get("generated")
            expected = ""
            if isinstance(generated, list) and generated:
                expected = str(generated[0])
            payload["item_id"] = item_id
            payload["expected"] = expected
            payload["metadata"] = {
                "rolebench_variant": variant_id,
                "subset": subset,
                "role": str(payload.get("role", "")),
            }
            normalized.append(payload)
        return CatalogNormalizedRows(rows=normalized)


def _load_rolebench_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    *,
    config_name: str | None = None,
    urlopen=request.urlopen,
) -> list[dict[str, object]]:
    if split != "test":
        raise ValueError("RoleBench builtin only supports the 'test' split.")
    variant_id = config_name or "instruction_generalization_eng"
    if variant_id not in _ROLEBENCH_VARIANT_PATHS:
        raise ValueError(f"Unsupported RoleBench variant: {variant_id}")
    desc_list = _load_json_file(
        dataset_id,
        revision,
        "profiles-eng/desc.json",
        urlopen=urlopen,
    )
    if not isinstance(desc_list, dict):
        raise ValueError("RoleBench desc.json must contain a JSON object.")
    rows: list[dict[str, object]] = []
    variant_path = _ROLEBENCH_VARIANT_PATHS[variant_id]
    for subset in _ROLEBENCH_SUBSETS:
        path = f"rolebench-eng/{variant_path}/{subset}/{split}.jsonl"
        for line_number, payload in enumerate(
            _load_jsonl_file(dataset_id, revision, path, urlopen=urlopen),
            start=1,
        ):
            role = str(payload.get("role", ""))
            payload["desc"] = str(desc_list.get(role, ""))
            payload["subset"] = subset
            payload["source_line_number"] = line_number
            rows.append(payload)
    return rows


def _load_json_file(
    dataset_id: str,
    revision: str | None,
    path: str,
    *,
    urlopen=request.urlopen,
) -> object:
    return load_json_url(
        _rolebench_file_url(dataset_id, revision, path),
        urlopen=urlopen,
        timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
    )


def _load_jsonl_file(
    dataset_id: str,
    revision: str | None,
    path: str,
    *,
    urlopen=request.urlopen,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in iter_jsonl_url(
        _rolebench_file_url(dataset_id, revision, path),
        urlopen=urlopen,
        timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
    ):
        if not isinstance(payload, dict):
            raise ValueError("RoleBench JSONL rows must be JSON objects.")
        rows.append(dict(payload))
    return rows


def _rolebench_file_url(dataset_id: str, revision: str | None, path: str) -> str:
    resolved_revision = revision or "main"
    return (
        f"https://huggingface.co/datasets/{dataset_id}/resolve/"
        f"{resolved_revision}/{path}"
    )


def _resolve_rolebench_variant_id(dataset: object) -> str:
    dimensions = getattr(dataset, "dimensions", None)
    if isinstance(dimensions, dict):
        variant_id = dimensions.get("rolebench_variant")
        if isinstance(variant_id, str) and variant_id:
            return variant_id
    config_name = getattr(getattr(dataset, "dataset", dataset), "config_name", None)
    if isinstance(config_name, str) and config_name:
        return config_name
    return "instruction_generalization_eng"
