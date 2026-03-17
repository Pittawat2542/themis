from __future__ import annotations

from types import SimpleNamespace

import pytest

from themis.errors import ThemisError
from themis.records.candidate import CandidateRecord
from themis.records.inference import InferenceRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import ErrorCode, DatasetSource


class _FakeSchemaValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _install_fake_jsonschema(monkeypatch: pytest.MonkeyPatch) -> None:
    def _validate_type(value, expected_type: str, path: str) -> None:
        if expected_type == "object" and not isinstance(value, dict):
            raise _FakeSchemaValidationError(f"{path} must be an object.")
        if expected_type == "array" and not isinstance(value, list):
            raise _FakeSchemaValidationError(f"{path} must be an array.")
        if expected_type == "string" and not isinstance(value, str):
            raise _FakeSchemaValidationError(f"{path} must be a string.")
        if expected_type == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise _FakeSchemaValidationError(f"{path} must be a number.")
        if expected_type == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise _FakeSchemaValidationError(f"{path} must be an integer.")
        if expected_type == "boolean" and not isinstance(value, bool):
            raise _FakeSchemaValidationError(f"{path} must be a boolean.")

    def _validate_schema(value, schema: dict[str, object], *, path: str = "$") -> None:
        expected_type = schema.get("type")
        if isinstance(expected_type, str):
            _validate_type(value, expected_type, path)

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            raise _FakeSchemaValidationError(f"{path} must be one of {enum_values!r}.")

        if expected_type == "object":
            required = schema.get("required", [])
            if isinstance(required, list):
                for key in required:
                    if isinstance(key, str) and (
                        not isinstance(value, dict) or key not in value
                    ):
                        raise _FakeSchemaValidationError(f"{path}.{key} is required.")
            properties = schema.get("properties", {})
            if isinstance(properties, dict) and isinstance(value, dict):
                for key, child_schema in properties.items():
                    if key not in value or not isinstance(child_schema, dict):
                        continue
                    _validate_schema(value[key], child_schema, path=f"{path}.{key}")

        if expected_type == "array":
            items_schema = schema.get("items")
            if isinstance(items_schema, dict) and isinstance(value, list):
                for index, item in enumerate(value):
                    _validate_schema(item, items_schema, path=f"{path}[{index}]")

    class _FakeValidator:
        def __init__(self, schema: dict[str, object]):
            self.schema = schema

        @classmethod
        def check_schema(cls, schema: object) -> None:
            if not isinstance(schema, dict):
                raise _FakeSchemaValidationError("schema must be an object")

        def validate(self, value: object) -> None:
            _validate_schema(value, self.schema)

    fake_jsonschema = SimpleNamespace(
        validators=SimpleNamespace(validator_for=lambda schema: _FakeValidator),
        exceptions=SimpleNamespace(
            ValidationError=_FakeSchemaValidationError,
            SchemaError=_FakeSchemaValidationError,
        ),
    )
    monkeypatch.setattr(
        "themis.extractors.builtin.import_optional",
        lambda module_name, *, extra: fake_jsonschema,
    )


def _trial() -> TrialSpec:
    return TrialSpec(
        trial_id="trial_extractors",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )


def _candidate(raw_text: str) -> CandidateRecord:
    return CandidateRecord(
        spec_hash="candidate_1",
        inference=InferenceRecord(spec_hash="inference_1", raw_text=raw_text),
    )


def test_registry_ships_builtin_extractors():
    registry = PluginRegistry()

    assert registry.has_extractor("regex")
    assert registry.has_extractor("json_schema")
    assert registry.has_extractor("first_number")
    assert registry.has_extractor("choice_letter")
    assert registry.has_extractor("boxed_text")
    assert registry.has_extractor("normalized_text")


def test_regex_extractor_parses_configured_capture_group():
    extractor = PluginRegistry().get_extractor("regex")
    extraction = extractor.extract(
        _trial(),
        _candidate("final score = 42"),
        {"pattern": r"score = (\d+)", "group": 1},
    )

    assert extraction.success is True
    assert extraction.parsed_answer == "42"


def test_json_schema_extractor_validates_simple_object_shape(monkeypatch):
    _install_fake_jsonschema(monkeypatch)
    extractor = PluginRegistry().get_extractor("json_schema")
    extraction = extractor.extract(
        _trial(),
        _candidate('{"answer": "42", "confidence": 0.9}'),
        {
            "schema": {
                "type": "object",
                "required": ["answer"],
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            }
        },
    )

    assert extraction.success is True
    assert extraction.parsed_answer == {"answer": "42", "confidence": 0.9}


def test_json_schema_extractor_rejects_bool_for_integer_schema(monkeypatch):
    _install_fake_jsonschema(monkeypatch)
    extractor = PluginRegistry().get_extractor("json_schema")
    extraction = extractor.extract(
        _trial(),
        _candidate("true"),
        {"schema": {"type": "integer"}},
    )

    assert extraction.success is False
    assert extraction.failure_reason is not None
    assert "integer" in extraction.failure_reason


def test_first_number_extractor_returns_numeric_value():
    extractor = PluginRegistry().get_extractor("first_number")
    extraction = extractor.extract(
        _trial(), _candidate("The result is -3.5 after rounding."), {}
    )

    assert extraction.success is True
    assert extraction.parsed_answer == -3.5


def test_choice_letter_extractor_returns_uppercase_choice():
    extractor = PluginRegistry().get_extractor("choice_letter")
    extraction = extractor.extract(_trial(), _candidate("I choose option c."), {})

    assert extraction.success is True
    assert extraction.parsed_answer == "C"


def test_choice_letter_extractor_handles_boxed_reasoning_answers():
    extractor = PluginRegistry().get_extractor("choice_letter")
    extraction = extractor.extract(
        _trial(),
        _candidate("After analysis, the final answer is \\boxed{b}."),
        {},
    )

    assert extraction.success is True
    assert extraction.parsed_answer == "B"


def test_boxed_text_extractor_returns_last_boxed_answer():
    extractor = PluginRegistry().get_extractor("boxed_text")
    extraction = extractor.extract(
        _trial(),
        _candidate("Scratch \\boxed{draft} and final \\boxed{42}"),
        {},
    )

    assert extraction.success is True
    assert extraction.parsed_answer == "42"


def test_normalized_text_extractor_cleans_whitespace_and_punctuation():
    extractor = PluginRegistry().get_extractor("normalized_text")
    extraction = extractor.extract(
        _trial(),
        _candidate("  The Answer!!!   "),
        {},
    )

    assert extraction.success is True
    assert extraction.parsed_answer == "the answer"


def test_json_schema_extractor_returns_install_hint_when_optional_dependency_is_missing(
    monkeypatch,
):
    extractor = PluginRegistry().get_extractor("json_schema")

    def raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(
        "themis.extractors.builtin.import_optional", raise_missing_optional
    )
    extraction = extractor.extract(
        _trial(), _candidate('{"answer": 42}'), {"schema": {"type": "object"}}
    )

    assert extraction.success is False
    assert (
        extraction.failure_reason
        == 'Install it with `uv add "themis-eval[extractors]"`.'
    )
