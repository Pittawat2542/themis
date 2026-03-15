"""Local filesystem blob-store implementation."""

from themis.storage.artifact_store import ArtifactStore


class LocalBlobStore(ArtifactStore):
    """Filesystem-backed content-addressed blob storage."""
