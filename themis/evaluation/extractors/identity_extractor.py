"""Identity (pass-through) extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IdentityExtractor:
    """Extractor that returns the raw output as-is.

    Args:
        strip_whitespace: Whether to strip leading/trailing whitespace
    """

    strip_whitespace: bool = True

    def extract(self, raw_output: str) -> str:
        """Return the raw output, optionally stripping whitespace.

        Args:
            raw_output: Raw output from model

        Returns:
            Raw output (possibly stripped)
        """
        if self.strip_whitespace:
            return raw_output.strip()
        return raw_output
