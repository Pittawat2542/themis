"""Helpers for integrating math-verify with Themis."""

from __future__ import annotations

import re
from typing import Any

from sympy import sympify

try:  # pragma: no cover - optional dependency
    from latex2sympy2_extended.math_normalization import NormalizationConfig
    from math_verify import (
        LatexExtractionConfig,
    )
    from math_verify import (
        parse as mv_parse,
    )
    from math_verify import (
        verify as mv_verify,
    )
except ImportError:  # pragma: no cover - triggered when math-verify isn't installed
    LatexExtractionConfig = None
    NormalizationConfig = None
    mv_parse = None
    mv_verify = None

_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def math_verify_available() -> bool:
    return mv_parse is not None and mv_verify is not None


def require_math_verify() -> None:
    if not math_verify_available():  # pragma: no cover - informative exception
        raise RuntimeError(
            "math-verify is required for math extraction/evaluation. Install via `uv pip install '.[math]'`."
        )


def extract_last_boxed(text: str) -> str:
    match = _BOXED_PATTERN.findall(text)
    if match:
        return match[-1]
    return text


def parse_expression(text: str) -> Any:
    require_math_verify()
    extraction_config = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(boxed="all"),
        )
    ]
    expressions = mv_parse(
        text,
        extraction_config=extraction_config,
        extraction_mode="first_match",
        fallback_mode="first_match",
    )
    expr = expressions[0] if expressions else text
    if isinstance(expr, str):
        try:
            return sympify(expr)
        except Exception:  # pragma: no cover - invalid sympy expr
            return expr
    return expr


def verify_expressions(reference: Any, prediction: Any) -> bool:
    require_math_verify()
    return bool(
        mv_verify(
            gold=reference,
            target=prediction,
            raise_on_error=False,
        )
    )


__all__ = [
    "math_verify_available",
    "require_math_verify",
    "extract_last_boxed",
    "parse_expression",
    "verify_expressions",
]
