"""Dataset provider for cais/hle."""

from __future__ import annotations

from ...datasets.common import (
    _HLE_RESPONSE_TEMPLATE,
    _apply_query,
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    load_hle_rows,
)


class BuiltinHLEDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_hle_rows)
        self._variant_scan_stats: dict[str, dict[str, object]] = {}

    def scan(self, slice_spec, query):
        dataset = slice_spec.dataset
        if dataset.dataset_id is None:
            raise ValueError(
                "Built-in HuggingFace dataset providers require a dataset_id."
            )
        rows = self.load_rows(dataset)
        normalized = self.prepare_rows(rows, slice_spec)
        filtered = _apply_query(normalized.rows, query)
        variant_id = _resolve_hle_variant_id(slice_spec)
        stats = {
            **normalized.stats,
            "loaded_count": len(rows),
            "normalized_count": len(normalized.rows),
            "returned_count": len(filtered),
        }
        self._variant_scan_stats[variant_id] = stats
        self._last_scan_stats = (
            dict(stats)
            if len(self._variant_scan_stats) == 1
            else {key: dict(value) for key, value in self._variant_scan_stats.items()}
        )
        return filtered

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        variant_id = _resolve_hle_variant_id(dataset)
        normalized: list[dict[str, object]] = []
        skipped = 0
        for row in rows:
            payload = dict(row)
            image = payload.get("image")
            if variant_id == "text_only" and isinstance(image, str) and image.strip():
                skipped += 1
                continue
            payload["item_id"] = str(payload.get("id", payload["item_id"]))
            metadata = {"hle_variant": variant_id}
            if variant_id == "text_only":
                metadata["text_only"] = "true"
            payload["metadata"] = metadata
            payload["expected_response"] = _HLE_RESPONSE_TEMPLATE.format(
                explanation="Demo benchmark answer.",
                answer=str(payload.get("answer", "")),
                confidence=100,
            )
            normalized.append(payload)
        return CatalogNormalizedRows(
            rows=normalized,
            stats={"skipped_image_count": skipped},
        )


def _resolve_hle_variant_id(dataset: object) -> str:
    dimensions = getattr(dataset, "dimensions", None)
    if isinstance(dimensions, dict):
        variant_id = dimensions.get("hle_variant")
        if isinstance(variant_id, str) and variant_id:
            return variant_id
    return "text_only"
