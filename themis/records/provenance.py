from typing import Dict
from pydantic import BaseModel, ConfigDict
from themis.types.hashable import HashableMixin


class ProvenanceRecord(HashableMixin, BaseModel):
    """
    Captures execution-environment metadata for reproducibility and debugging.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    themis_version: str
    git_commit: str | None
    python_version: str
    platform: str
    library_versions: Dict[str, str]
    model_endpoint_meta: Dict[str, str]
