"""Dataset provider for m-a-p/SuperGPQA."""

from __future__ import annotations

from .common import BuiltinMCQDatasetProvider


class BuiltinSuperGPQADatasetProvider(BuiltinMCQDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(
            metadata_keys=["discipline", "field", "subfield", "difficulty"],
            huggingface_loader=huggingface_loader,
        )
