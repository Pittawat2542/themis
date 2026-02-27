"""Regex-based extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .exceptions import FieldExtractionError


@dataclass
class RegexExtractor:
    """Extracts fields using regular expression patterns.

    Args:
        pattern: Regular expression pattern with optional named groups
    """

    pattern: str

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern)

    def extract(self, text: str) -> dict[str, str]:
        """Extract fields from text using regex pattern.

        Args:
            text: Text to extract from

        Returns:
            Dictionary of extracted groups (named or numbered)

        Raises:
            FieldExtractionError: If pattern does not match
        """
        match = self._compiled.search(text)
        if not match:
            raise FieldExtractionError("Regex did not match")
        groups = match.groupdict()
        if groups:
            return {
                key: value.strip() for key, value in groups.items() if value is not None
            }
        return {
            str(index): value.strip()
            for index, value in enumerate(match.groups())
            if value is not None
        }
