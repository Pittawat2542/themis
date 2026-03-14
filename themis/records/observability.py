from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ObservabilityLink(BaseModel):
    """One provider-specific observability link attached to a trial or candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    external_id: str | None = None
    url: str | None = None
    extras: dict[str, str] = Field(default_factory=dict)


class ObservabilitySnapshot(BaseModel):
    """Projection-side observability links that are not hashed into domain records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    links: list[ObservabilityLink] = Field(default_factory=list)

    def link_for(self, provider: str) -> ObservabilityLink | None:
        """Return the first link for one provider."""
        for link in self.links:
            if link.provider == provider:
                return link
        return None

    def external_id_for(self, provider: str) -> str | None:
        """Return one provider-specific external identifier."""
        link = self.link_for(provider)
        return link.external_id if link is not None else None

    def url_for(self, provider: str) -> str | None:
        """Return one provider-specific URL."""
        link = self.link_for(provider)
        return link.url if link is not None else None

    @property
    def langfuse_trace_id(self) -> str | None:
        return self.external_id_for("langfuse")

    @property
    def langfuse_url(self) -> str | None:
        return self.url_for("langfuse")

    @property
    def wandb_url(self) -> str | None:
        return self.url_for("wandb")


ObservabilityRefs = ObservabilitySnapshot
