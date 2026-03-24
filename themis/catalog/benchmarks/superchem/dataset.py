"""Dataset provider for ZehuaZhao/SUPERChem."""

from __future__ import annotations

from ...datasets._loaders import _datasets_load_dataset
from ...datasets._normalizers import _normalize_superchem_rows
from ...datasets._providers import BuiltinDatasetProvider, _assign_missing_item_ids
from ...datasets._types import CatalogNormalizedRows


class BuiltinSuperChemDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or _load_superchem_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_superchem_rows(rows, dataset)


def _load_superchem_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[dict[str, object]]:
    from themis._optional import import_optional

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    dataset = _datasets_load_dataset(
        datasets,
        dataset_id,
        split=split,
        revision=revision,
        config_name=config_name,
        streaming=True,
    )
    return _assign_missing_item_ids([dict(row) for row in dataset])
