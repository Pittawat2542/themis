"""Builtin parsers."""

from __future__ import annotations

import json
import re
from typing import Any

from themis.core.base import JSONValue
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


class RegexParser:
    component_id = "builtin/regex"
    version = "1.0"

    def __init__(
        self,
        *,
        pattern: str,
        group: int | str | None = None,
        ignore_case: bool = False,
        multiline: bool = False,
        dot_all: bool = False,
    ) -> None:
        flags = 0
        if ignore_case:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dot_all:
            flags |= re.DOTALL
        self.pattern = pattern
        self.group = group
        self.ignore_case = ignore_case
        self.multiline = multiline
        self.dot_all = dot_all
        self._compiled = re.compile(pattern, flags)

    def fingerprint(self) -> str:
        return (
            "builtin-regex-fingerprint:"
            f"{self.pattern}:{self.group}:{self.ignore_case}:{self.multiline}:{self.dot_all}"
        )

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        raw_text = _stringify(candidate.final_output)
        match = self._compiled.search(raw_text)
        if match is None:
            raise ValueError("Regex did not match candidate output")
        try:
            value = (
                match.group(self.group) if self.group is not None else match.group(0)
            )
        except (IndexError, KeyError) as exc:
            raise ValueError(f"Regex group is not available: {self.group}") from exc
        metadata: dict[str, JSONValue] = {}
        if self.group is not None:
            metadata["group"] = self.group
        return ParsedOutput(value=value, format="regex", metadata=metadata)


class SchemaParser:
    component_id = "builtin/schema"
    version = "1.0"

    def __init__(
        self,
        *,
        schema: dict[str, JSONValue],
        path: str | None = None,
    ) -> None:
        self.schema = schema
        self.path = path

    def fingerprint(self) -> str:
        schema_json = json.dumps(self.schema, sort_keys=True, separators=(",", ":"))
        return f"builtin-schema-fingerprint:{schema_json}:{self.path}"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        value = _load_json_like(candidate.final_output)
        _validate_schema(value, self.schema, path="$")
        output = _extract_path(value, self.path) if self.path is not None else value
        metadata: dict[str, JSONValue] = {}
        if self.path is not None:
            metadata["path"] = self.path
        return ParsedOutput(
            value=_as_json_value(output), format="schema", metadata=metadata
        )


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


def _load_json_like(value: JSONValue) -> JSONValue:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    return _as_json_value(parsed)


def _validate_schema(
    value: JSONValue, schema: dict[str, JSONValue], *, path: str
) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        _validate_type(value, expected_type, path=path)

    properties = schema.get("properties")
    required = schema.get("required")
    if isinstance(required, list):
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}")
        for field in required:
            if not isinstance(field, str):
                raise ValueError(f"Schema required fields must be strings at {path}")
            if field not in value:
                raise ValueError(f"Missing required field at {path}.{field}")
    if isinstance(properties, dict):
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}")
        for key, child_schema in properties.items():
            if key in value:
                if not isinstance(child_schema, dict):
                    raise ValueError(
                        f"Schema property must be an object at {path}.{key}"
                    )
                _validate_schema(
                    value[key],
                    _schema_dict(child_schema),
                    path=f"{path}.{key}",
                )

    items = schema.get("items")
    if isinstance(items, dict):
        if not isinstance(value, list):
            raise ValueError(f"Expected array at {path}")
        for index, item in enumerate(value):
            _validate_schema(item, _schema_dict(items), path=f"{path}[{index}]")


def _validate_type(value: JSONValue, expected_type: str, *, path: str) -> None:
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}")
        return
    if expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"Expected array at {path}")
        return
    if expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"Expected string at {path}")
        return
    if expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Expected number at {path}")
        return
    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Expected integer at {path}")
        return
    if expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"Expected boolean at {path}")
        return
    if expected_type == "null":
        if value is not None:
            raise ValueError(f"Expected null at {path}")
        return
    raise ValueError(f"Unsupported schema type at {path}: {expected_type}")


def _extract_path(value: JSONValue, path: str | None) -> JSONValue:
    if path is None:
        return value
    current: JSONValue = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise ValueError(f"Schema path is not available: {path}")
    return current


def _schema_dict(value: dict[Any, Any]) -> dict[str, JSONValue]:
    return {str(key): _as_json_value(item) for key, item in value.items()}


def _as_json_value(value: Any) -> JSONValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_as_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    raise ValueError(f"Value is not JSON-compatible: {type(value).__name__}")
