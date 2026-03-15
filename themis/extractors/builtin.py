"""Built-in extractor implementations auto-registered by `PluginRegistry`."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from themis._optional import import_optional
from themis.errors import ThemisError
from themis.records.candidate import CandidateRecord
from themis.records.extraction import ExtractionRecord
from themis.specs.experiment import TrialSpec
from themis.types.json_types import JSONValueType


def _extraction_spec_hash(
    extractor_id: str, candidate: CandidateRecord, config: Mapping[str, JSONValueType]
) -> str:
    payload = json.dumps(
        {
            "extractor_id": extractor_id,
            "candidate_id": candidate.spec_hash,
            "config": dict(config),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _raw_text(candidate: CandidateRecord) -> str:
    if candidate.inference is None or candidate.inference.raw_text is None:
        return ""
    return candidate.inference.raw_text


def _success(
    extractor_id: str,
    candidate: CandidateRecord,
    config: Mapping[str, JSONValueType],
    parsed_answer: JSONValueType,
) -> ExtractionRecord:
    return ExtractionRecord(
        spec_hash=_extraction_spec_hash(extractor_id, candidate, config),
        extractor_id=extractor_id,
        success=True,
        parsed_answer=parsed_answer,
    )


def _failure(
    extractor_id: str,
    candidate: CandidateRecord,
    config: Mapping[str, JSONValueType],
    failure_reason: str,
) -> ExtractionRecord:
    return ExtractionRecord(
        spec_hash=_extraction_spec_hash(extractor_id, candidate, config),
        extractor_id=extractor_id,
        success=False,
        failure_reason=failure_reason,
    )


class RegexExtractor:
    """Extract a regex match or capture group from candidate raw text."""

    def extract(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        config: Mapping[str, JSONValueType] | None = None,
    ) -> ExtractionRecord:
        """Extract a configured regex match from candidate output."""
        del trial
        cfg = dict(config or {})
        pattern = cfg.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return _failure(
                "regex",
                candidate,
                cfg,
                "Regex extractor requires a non-empty 'pattern'.",
            )

        group = cfg.get("group", 0)
        text = _raw_text(candidate)
        match = re.search(pattern, text)
        if match is None:
            return _failure(
                "regex", candidate, cfg, "Pattern did not match the inference output."
            )

        if not isinstance(group, (int, str)):
            return _failure(
                "regex",
                candidate,
                cfg,
                "Configured capture group must be an integer or string.",
            )

        try:
            value = match.group(int(group))
        except (IndexError, ValueError, TypeError):
            return _failure(
                "regex",
                candidate,
                cfg,
                "Configured capture group was not present in the regex match.",
            )
        return _success("regex", candidate, cfg, value)


class JsonSchemaExtractor:
    """Parse candidate raw text as JSON and validate it against a schema."""

    def extract(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        config: Mapping[str, JSONValueType] | None = None,
    ) -> ExtractionRecord:
        """Parse candidate output as JSON and validate it against a schema."""
        del trial
        cfg = dict(config or {})
        schema = cfg.get("schema")
        if not isinstance(schema, Mapping):
            return _failure(
                "json_schema",
                candidate,
                cfg,
                "Json schema extractor requires a 'schema' object.",
            )

        try:
            parsed = json.loads(_raw_text(candidate))
        except json.JSONDecodeError as exc:
            return _failure(
                "json_schema",
                candidate,
                cfg,
                f"Response was not valid JSON: {exc.msg}.",
            )

        try:
            jsonschema = import_optional("jsonschema", extra="extractors")
        except ThemisError as exc:
            return _failure("json_schema", candidate, cfg, exc.message)

        try:
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            validator_cls(schema).validate(parsed)
        except jsonschema.exceptions.SchemaError as exc:
            return _failure(
                "json_schema", candidate, cfg, f"Invalid JSON schema: {exc.message}"
            )
        except jsonschema.exceptions.ValidationError as exc:
            return _failure("json_schema", candidate, cfg, exc.message)
        return _success("json_schema", candidate, cfg, parsed)


class FirstNumberExtractor:
    """Extract the first integer or floating-point token from raw text."""

    def extract(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        config: Mapping[str, JSONValueType] | None = None,
    ) -> ExtractionRecord:
        """Extract the first numeric token from candidate output."""
        del trial
        cfg = dict(config or {})
        text = _raw_text(candidate)
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match is None:
            return _failure(
                "first_number",
                candidate,
                cfg,
                "No numeric token found in the inference output.",
            )
        number_text = match.group(0)
        parsed: int | float = (
            int(number_text) if "." not in number_text else float(number_text)
        )
        return _success("first_number", candidate, cfg, parsed)


class ChoiceLetterExtractor:
    """Extract an uppercase multiple-choice letter from candidate raw text."""

    def extract(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        config: Mapping[str, JSONValueType] | None = None,
    ) -> ExtractionRecord:
        """Extract a multiple-choice answer letter from candidate output."""
        del trial
        cfg = dict(config or {})
        configured_choices = cfg.get("choices")
        if isinstance(configured_choices, list) and configured_choices:
            choices = [str(choice).upper() for choice in configured_choices]
        else:
            choices = ["A", "B", "C", "D", "E"]
        choices_pattern = "".join(re.escape(choice) for choice in choices)
        text = _raw_text(candidate)

        match = re.search(
            rf"\b(?:option|answer|choice)\s*[:\-]?\s*([{choices_pattern}])\b",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            match = re.search(rf"\b([{choices_pattern}])\b", text, flags=re.IGNORECASE)
        if match is None:
            return _failure(
                "choice_letter",
                candidate,
                cfg,
                "No choice letter found in the inference output.",
            )
        return _success("choice_letter", candidate, cfg, match.group(1).upper())
