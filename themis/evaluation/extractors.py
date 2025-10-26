"""Output extractors used during evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict


class FieldExtractionError(RuntimeError):
    """Raised when an output field cannot be extracted."""


@dataclass
class JsonFieldExtractor:
    field_path: str

    def extract(self, raw_output: str) -> Any:
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


@dataclass
class RegexExtractor:
    pattern: str

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern)

    def extract(self, text: str) -> Dict[str, str]:
        match = self._compiled.search(text)
        if not match:
            raise FieldExtractionError("Regex did not match")
        groups = match.groupdict()
        if groups:
            return {key: value.strip() for key, value in groups.items()}
        return {str(index): value.strip() for index, value in enumerate(match.groups())}


@dataclass
class MathVerifyExtractor:
    """Extracts the final boxed answer using math-verify parsing."""

    def extract(self, raw_output: str) -> str:
        from themis.evaluation import math_verify_utils as mv_utils

        candidate = mv_utils.extract_last_boxed(raw_output)
        try:
            parsed = mv_utils.parse_expression(candidate)
        except Exception as exc:  # pragma: no cover - parse failure
            raise FieldExtractionError("math-verify parsing failed") from exc
        return str(parsed).strip()
