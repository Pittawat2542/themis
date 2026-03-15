from pydantic import BaseModel, ConfigDict, Field

from themis.types.hashable import HashableMixin, SHORT_HASH_LENGTH


class SpecBase(HashableMixin, BaseModel):
    """
    Base class for immutable, canonically hashable input configuration models.

    All spec models are frozen, strict, and excluded from unknown fields so their
    serialized form stays stable across planning, storage, and replay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: str = Field(
        default="1.0",
        description="Version string for migrations.",
        json_schema_extra={"exclude_from_hash": True},
    )

    @property
    def canonical_hash(self) -> str:
        """Returns the full canonical hash for internal identity checks."""
        return self.compute_hash()

    @property
    def spec_hash(self) -> str:
        """Returns the 12-character public alias for this spec identity."""
        return self.canonical_hash[:SHORT_HASH_LENGTH]
