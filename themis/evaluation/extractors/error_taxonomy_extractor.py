from __future__ import annotations

import re
from typing import Dict


class ErrorTaxonomyExtractor:
    """
    Lightweight error taxonomy extractor.

    Heuristics:
    - format_parse_failure: Unbalanced JSON-like braces suggest parsing intent but malformed format
    - arithmetic_slip: Simple arithmetic expression like "X + Y = Z" evaluated incorrectly
    - reasoning_gap: Final answer given without common justification keywords
    """

    def extract(self, text: str) -> Dict[str, bool]:
        labels = {
            "format_parse_failure": False,
            "arithmetic_slip": False,
            "reasoning_gap": False,
        }

        # format_parse_failure: JSON-like but malformed braces
        if ("{" in text or "}" in text) and not self._balanced_braces(text):
            labels["format_parse_failure"] = True

        # arithmetic_slip: pattern "A op B = Z" mismatch
        try:
            if self._has_arithmetic_mismatch(text):
                labels["arithmetic_slip"] = True
        except Exception:
            pass

        # reasoning_gap: answer provided without justification keywords
        lowered = text.lower()
        has_answer_phrase = any(p in lowered for p in ("answer", "final", "therefore"))
        has_justification = any(
            k in lowered for k in ("because", "since", "thus", "therefore", "reason")
        )
        if has_answer_phrase and not has_justification:
            labels["reasoning_gap"] = True

        return labels

    def _balanced_braces(self, text: str) -> bool:
        count = 0
        for ch in text:
            if ch == "{":
                count += 1
            elif ch == "}":
                count -= 1
                if count < 0:
                    return False
        return count == 0

    def _has_arithmetic_mismatch(self, text: str) -> bool:
        pattern = (
            r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
        )
        m = re.search(pattern, text)
        if not m:
            return False
        a, op, b, z = m.groups()
        a = float(a)
        b = float(b)
        z = float(z)
        if op == "+":
            calc = a + b
        elif op == "-":
            calc = a - b
        elif op == "*":
            calc = a * b
        elif op == "/":
            if b == 0:
                return False
            calc = a / b
        else:
            return False
        return abs(calc - z) > 1e-9
