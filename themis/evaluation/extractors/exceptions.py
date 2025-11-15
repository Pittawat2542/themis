"""Extractor exceptions."""

from __future__ import annotations


class FieldExtractionError(RuntimeError):
    """Raised when an output field cannot be extracted."""
