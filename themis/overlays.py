"""Shared overlay selection helpers for generation, transform, and evaluation views."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OverlaySelection:
    """A validated overlay selection that can be converted to a storage key."""

    transform_hash: str | None = None
    evaluation_hash: str | None = None

    def __post_init__(self) -> None:
        if self.evaluation_hash is not None and self.evaluation_hash == "":
            raise ValueError("OverlaySelection.evaluation_hash cannot be empty.")
        if self.transform_hash is not None and self.transform_hash == "":
            raise ValueError("OverlaySelection.transform_hash cannot be empty.")

    @property
    def overlay_key(self) -> str:
        """Return the storage key for the selected overlay."""
        if self.evaluation_hash is not None:
            return f"ev:{self.evaluation_hash}"
        if self.transform_hash is not None:
            return f"tf:{self.transform_hash}"
        return "gen"

    def metadata(self) -> dict[str, str]:
        """Return report-friendly metadata for this overlay selection."""
        metadata = {"overlay_key": self.overlay_key}
        if self.transform_hash is not None:
            metadata["transform_hash"] = self.transform_hash
        if self.evaluation_hash is not None:
            metadata["evaluation_hash"] = self.evaluation_hash
        return metadata


def overlay_key_for(
    *,
    transform_hash: str | None = None,
    evaluation_hash: str | None = None,
) -> str:
    """Return the canonical storage overlay key for the given selection."""
    return OverlaySelection(
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    ).overlay_key
