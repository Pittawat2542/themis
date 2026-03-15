from themis.records.inference import InferenceRecord, TokenUsage
from themis.types.enums import RecordStatus


def test_inference_record():
    record = InferenceRecord(
        spec_hash="xyz",
        status=RecordStatus.OK,
        raw_text="The answer is 42.",
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=6, total_tokens=16),
        latency_ms=150,
        structured_output={"answer": 42},
    )

    assert record.raw_text == "The answer is 42."
    assert record.token_usage.total_tokens == 16
    assert record.structured_output["answer"] == 42
    assert record.logprobs is None
