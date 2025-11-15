"""Math-verify extraction for mathematical expressions."""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import FieldExtractionError


@dataclass
class MathVerifyExtractor:
    """Extracts the final boxed answer using math-verify parsing.

    This extractor uses the math-verify library to parse and normalize
    mathematical expressions from LaTeX boxed notation.
    """

    def extract(self, raw_output: str) -> str:
        """Extract and parse boxed mathematical answer.

        Args:
            raw_output: Raw output containing \\boxed{...} notation

        Returns:
            Parsed and normalized mathematical expression

        Raises:
            FieldExtractionError: If math-verify parsing fails
        """
        from themis.evaluation import math_verify_utils as mv_utils

        candidate = mv_utils.extract_last_boxed(raw_output)
        try:
            parsed = mv_utils.parse_expression(candidate)
        except Exception as exc:  # pragma: no cover - parse failure
            raise FieldExtractionError("math-verify parsing failed") from exc
        return str(parsed).strip()
