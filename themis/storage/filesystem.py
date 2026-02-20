"""Filesystem operations for storage."""

from __future__ import annotations

import gzip
import json
import os

import tempfile
from pathlib import Path
from collections.abc import Iterable
from typing import TextIO

from themis.storage.models import StorageConfig

STORAGE_FORMAT_VERSION = "2.0.0"


class FileSystem:
    """Manages filesystem operations with atomic writes and compression."""

    def __init__(self, config: StorageConfig) -> None:
        self.config = config

    def file_exists_any_compression(self, path: Path) -> bool:
        """Check if file exists with any compression suffix."""
        return path.exists() or path.with_suffix(path.suffix + ".gz").exists()

    def open_for_read(self, path: Path) -> TextIO:
        """Open file for reading with automatic compression detection.

        Args:
            path: File path

        Returns:
            File handle (text mode)
        """
        # Try .gz version first
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            return gzip.open(gz_path, "rt", encoding="utf-8")
        if path.exists():
            return path.open("r", encoding="utf-8")
        raise FileNotFoundError(f"File not found: {path}")

    def write_jsonl_with_header(
        self, path: Path, items: Iterable[dict], file_type: str
    ) -> None:
        """Write JSONL file with format version header."""
        # Determine actual path based on compression
        if self.config.compression == "gzip":
            actual_path = path.with_suffix(path.suffix + ".gz")
            handle = gzip.open(actual_path, "wt", encoding="utf-8")
        else:
            actual_path = path
            handle = open(actual_path, "w", encoding="utf-8")

        with handle:
            # Write header
            header = {
                "_type": "header",
                "_format_version": STORAGE_FORMAT_VERSION,
                "_file_type": file_type,
            }
            handle.write(json.dumps(header) + "\n")

            # Write items
            for item in items:
                handle.write(json.dumps(item) + "\n")

            handle.flush()
            if hasattr(handle, "fileno"):
                os.fsync(handle.fileno())

    def atomic_append(self, path: Path, data: dict) -> None:
        """Append data atomically using temp file.

        Args:
            path: Target file path
            data: Data to append (will be JSON serialized)
        """
        json_line = json.dumps(data) + "\n"

        # Write to temp file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".json"
        )
        temp_path = Path(temp_path)

        try:
            if self.config.compression == "gzip":
                # Close the fd first since gzip.open will open by path
                os.close(temp_fd)
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    f.write(json_line)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                # Use the fd directly
                with open(temp_fd, "w", encoding="utf-8") as f:
                    f.write(json_line)
                    f.flush()
                    os.fsync(f.fileno())
                # fd is closed by context manager, don't close again

            # Get target path with compression
            target_path = (
                path.with_suffix(path.suffix + ".gz")
                if self.config.compression == "gzip"
                else path
            )

            # Append to existing file
            if target_path.exists():
                with open(target_path, "ab") as dest:
                    with open(temp_path, "rb") as src:
                        dest.write(src.read())
                    dest.flush()
                    os.fsync(dest.fileno())
            else:
                # No existing file, just rename
                temp_path.rename(target_path)
                return

        finally:
            # Clean up temp file if still exists
            if temp_path.exists():
                temp_path.unlink()
