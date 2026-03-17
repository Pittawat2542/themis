"""Benchmark-native dataset query models."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from themis.specs.base import SpecBase
from themis.types.enums import SamplingKind


class DatasetQuerySpec(SpecBase):
    """Declarative slice query and sampling controls for dataset providers."""

    kind: SamplingKind = Field(default=SamplingKind.ALL)
    count: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None)
    strata_field: str | None = Field(default=None)
    item_ids: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, str] = Field(default_factory=dict)
    projected_fields: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: SamplingKind | str) -> SamplingKind | str:
        if isinstance(value, str):
            return SamplingKind(value)
        return value

    @classmethod
    def all(cls) -> "DatasetQuerySpec":
        return cls(kind=SamplingKind.ALL)

    @classmethod
    def subset(
        cls,
        count: int,
        *,
        seed: int | None = None,
    ) -> "DatasetQuerySpec":
        return cls(kind=SamplingKind.SUBSET, count=count, seed=seed)

    @classmethod
    def stratified(
        cls,
        count: int,
        *,
        strata_field: str,
        seed: int | None = None,
    ) -> "DatasetQuerySpec":
        return cls(
            kind=SamplingKind.STRATIFIED,
            count=count,
            seed=seed,
            strata_field=strata_field,
        )

    @model_validator(mode="after")
    def _validate_semantic(self) -> "DatasetQuerySpec":
        if (
            self.kind in {SamplingKind.SUBSET, SamplingKind.STRATIFIED}
            and self.count is None
        ):
            raise ValueError(
                f"DatasetQuerySpec kind='{self.kind.value}' requires a positive count."
            )
        if self.kind == SamplingKind.STRATIFIED and not self.strata_field:
            raise ValueError(
                "DatasetQuerySpec kind='stratified' requires strata_field."
            )
        return self
