"""Content-addressed artifact persistence for prompts, payloads, and audits."""

import hashlib
import json
from pathlib import Path

from themis._optional import import_optional
from themis.errors.exceptions import StorageError
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import ErrorCode
from themis.types.json_types import JSONValueType
from themis.types.json_validation import dump_storage_json_bytes


class ArtifactStore:
    """
    Content-addressed blob storage with optional zstd compression.

    Artifacts are sharded by the first two hex pairs of their SHA-256 hash so
    large prompt, payload, and audit blobs can be deduplicated on disk.
    """

    def __init__(self, base_path: Path, manager: DatabaseManager | None = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.manager = manager
        zstd = import_optional("zstandard", extra="compression")
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()

    def _get_path(self, hex_hash: str) -> Path:
        """Returns the sharded path: base/ab/cd/abcdef....zst"""
        d1, d2 = hex_hash[:2], hex_hash[2:4]
        return self.base_path / d1 / d2 / f"{hex_hash}.zst"

    def put_blob(self, blob: bytes, media_type: str) -> str:
        """Persist one raw artifact blob and return its stable hash reference."""
        try:
            hex_hash = hashlib.sha256(blob).hexdigest()
            target_path = self._get_path(hex_hash)
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                compressed = self.compressor.compress(blob)
                target_path.write_bytes(compressed)
            artifact_ref = f"sha256:{hex_hash}"
            self._index_artifact(
                artifact_ref,
                target_path,
                media_type=media_type,
                size_bytes=len(blob),
            )
            return artifact_ref
        except Exception as e:
            raise StorageError(
                code=ErrorCode.STORAGE_WRITE,
                message=f"Failed to write artifact: {str(e)}",
            ) from e

    def get_blob(self, ref: str) -> bytes:
        """Load one artifact blob by its `sha256:` reference."""
        if not ref.startswith("sha256:"):
            raise StorageError(
                code=ErrorCode.STORAGE_READ, message=f"Invalid hash format: {ref}"
            )

        hex_hash = ref[7:]
        target_path = self._get_path(hex_hash)

        if not target_path.exists():
            raise StorageError(
                code=ErrorCode.STORAGE_READ, message=f"Artifact {ref} not found."
            )

        try:
            compressed = target_path.read_bytes()
            return self.decompressor.decompress(compressed)
        except Exception as e:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=f"Failed to read artifact: {str(e)}",
            ) from e

    def write_json(self, data: JSONValueType) -> tuple[str, str]:
        """
        Serializes to JSON, computes SHA-256, and writes compressed chunk.
        Returns (file_uri, sha256_hash).
        """
        payload = dump_storage_json_bytes(data, label="application/json payload")
        blob_ref = self.put_blob(payload, "application/json")
        file_uri = f"file://{self._get_path(blob_ref[7:]).absolute()}"
        return file_uri, blob_ref

    def read_json(self, sha256_hash: str) -> JSONValueType:
        """
        Reads and decompresses a JSON artifact by its hash.
        """
        payload = self.get_blob(sha256_hash)
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise StorageError(
                code=ErrorCode.STORAGE_READ,
                message=f"Failed to decode JSON artifact {sha256_hash}: {exc.msg}",
            ) from exc

    def exists(self, sha256_hash: str) -> bool:
        """Check if an artifact exists by its hash."""
        if not sha256_hash.startswith("sha256:"):
            return False
        return self._get_path(sha256_hash[7:]).exists()

    def _index_artifact(
        self,
        artifact_hash: str,
        path: Path,
        *,
        media_type: str,
        size_bytes: int,
    ) -> None:
        if self.manager is None:
            return
        with self.manager.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO artifacts (artifact_hash, path, size_bytes, compression, media_type)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_hash) DO UPDATE SET
                        path=excluded.path,
                        size_bytes=excluded.size_bytes,
                        compression=excluded.compression,
                        media_type=excluded.media_type
                    """,
                    (
                        artifact_hash,
                        str(path),
                        size_bytes,
                        "zstd",
                        media_type,
                    ),
                )
