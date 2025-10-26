import pytest

from themis.evaluation import extractors


def test_json_field_extractor_reads_nested_fields():
    extractor = extractors.JsonFieldExtractor(field_path="answer.value")
    raw_output = '{"answer": {"value": "Paris"}, "score": 0.85}'

    value = extractor.extract(raw_output)

    assert value == "Paris"


def test_json_field_extractor_raises_clear_error_on_missing_field():
    extractor = extractors.JsonFieldExtractor(field_path="answer.value")

    with pytest.raises(extractors.FieldExtractionError) as exc:
        extractor.extract('{"answer": {"label": "Paris"}}')

    assert "answer.value" in str(exc.value)


def test_regex_extractor_can_capture_multiple_groups():
    extractor = extractors.RegexExtractor(
        pattern=r"Answer:(?P<answer>.+)Score:(?P<score>.+)"
    )
    text = "Answer:Paris Score:0.92"

    extracted = extractor.extract(text)

    assert extracted["answer"] == "Paris"
    assert extracted["score"] == "0.92"
