"""Dataset provider for TIGER-Lab/MMLU-Pro."""

from __future__ import annotations

from .common import BuiltinMCQDatasetProvider


class BuiltinMMLUProDatasetProvider(BuiltinMCQDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(
            metadata_keys=["category", "src"],
            huggingface_loader=huggingface_loader,
        )
