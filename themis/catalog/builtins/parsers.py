"""Builtin parsers."""

from __future__ import annotations

import re

from themis.core.contexts import ParseContext
from themis.core.models import ParsedOutput, ReducedCandidate


class JsonIdentityParser:
    component_id = "builtin/json_identity"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-json-identity-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        return ParsedOutput(value=candidate.final_output, format="json")


class TextParser:
    component_id = "builtin/text"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-text-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        value = candidate.final_output
        if isinstance(value, str):
            return ParsedOutput(value=value, format="text")
        return ParsedOutput(value=str(value), format="text")


class ChoiceLetterParser:
    component_id = "builtin/choice_letter"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-choice-letter-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        raw_text = _stringify(candidate.final_output)
        matches = re.findall(r"\b([A-J])\b", raw_text.upper())
        value = matches[-1] if matches else ""
        return ParsedOutput(value=value, format="choice_letter")


class MathAnswerParser:
    component_id = "builtin/math_answer"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-math-answer-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        raw_text = _stringify(candidate.final_output)
        boxed_match = re.findall(r"\\boxed\{([^{}]+)\}", raw_text)
        if boxed_match:
            return ParsedOutput(value=boxed_match[-1].strip(), format="math_answer")
        return ParsedOutput(value=raw_text.strip(), format="math_answer")


class CodeTextParser:
    component_id = "builtin/code_text"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-code-text-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        value = candidate.final_output
        if isinstance(value, dict):
            for key in ("solution", "code", "answer"):
                candidate_value = value.get(key)
                if isinstance(candidate_value, str) and candidate_value.strip():
                    return ParsedOutput(
                        value=_extract_code_block(candidate_value),
                        format="code",
                    )
        return ParsedOutput(value=_extract_code_block(_stringify(value)), format="code")


def _stringify(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _extract_code_block(text: str) -> str:
    pattern = re.compile(
        r"```(?:python|py|cpp|c\+\+)?\s*(?P<code>.+?)\s*```",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group("code").strip()
    return text.strip()
