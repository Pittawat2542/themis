"""JSON field extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .exceptions import FieldExtractionError


@dataclass
class JsonFieldExtractor:
    """Extracts a specific field from JSON output.

    Args:
        field_path: Dot-separated path to the field (e.g., "answer" or "result.value")
    """

    field_path: str

    def extract(self, raw_output: str) -> Any:
        """Extract the specified field from JSON output.

        Args:
            raw_output: Raw JSON string from model

        Returns:
            Extracted field value

        Raises:
            FieldExtractionError: If JSON is invalid or field is missing
        """
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
            raise FieldExtractionError("Invalid JSON output") from exc

        current = payload
        for part in self.field_path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise FieldExtractionError(f"Missing field '{self.field_path}'")
        return current
