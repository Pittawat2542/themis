"""Dataset provider for m-a-p/Encyclo-K."""

from __future__ import annotations

from .common import BuiltinMCQDatasetProvider


class BuiltinEncycloKDatasetProvider(BuiltinMCQDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(
            metadata_keys=["discipline", "field", "subfield", "difficulty"],
            huggingface_loader=huggingface_loader,
        )
