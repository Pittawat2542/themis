from themis.types.enums import (
    DatasetSource,
    CompressionCodec,
    ErrorCode,
    ErrorWhere,
    InferenceStatus,
    IssueSeverity,
    PValueCorrection,
    RecordStatus,
    RecordType,
    ResponseFormat,
    SamplingKind,
    StorageBackend,
)
from themis.types.events import ArtifactRole, TimelineStage, TrialEventType


def test_record_status_values():
    assert set(RecordStatus) == {
        RecordStatus.OK,
        RecordStatus.ERROR,
        RecordStatus.SKIPPED,
        RecordStatus.PARTIAL,
    }
    assert RecordStatus.OK.value == "ok"
    assert RecordStatus.ERROR.value == "error"


def test_inference_status_values():
    assert set(InferenceStatus) == {
        InferenceStatus.OK,
        InferenceStatus.ERROR,
    }


def test_error_where_values():
    assert set(ErrorWhere) == {
        ErrorWhere.PLANNER,
        ErrorWhere.EXECUTOR,
        ErrorWhere.INFERENCE,
        ErrorWhere.EXTRACTOR,
        ErrorWhere.METRIC,
        ErrorWhere.STORAGE,
    }


def test_error_code_values():
    expected_codes = {
        "provider_timeout",
        "provider_auth",
        "provider_rate_limit",
        "provider_unavailable",
        "parse_error",
        "schema_mismatch",
        "metric_computation",
        "storage_write",
        "storage_read",
        "plugin_incompatible",
        "missing_optional_dependency",
        "circuit_breaker",
        "item_load",
    }
    actual_codes = {e.value for e in ErrorCode}
    assert actual_codes == expected_codes


def test_trial_event_type_values():
    assert set(TrialEventType) == {
        TrialEventType.TRIAL_STARTED,
        TrialEventType.ITEM_LOADED,
        TrialEventType.PROMPT_RENDERED,
        TrialEventType.CANDIDATE_STARTED,
        TrialEventType.CONVERSATION_EVENT,
        TrialEventType.INFERENCE_COMPLETED,
        TrialEventType.EXTRACTION_COMPLETED,
        TrialEventType.EVALUATION_COMPLETED,
        TrialEventType.CANDIDATE_COMPLETED,
        TrialEventType.CANDIDATE_FAILED,
        TrialEventType.TRIAL_RETRY,
        TrialEventType.PROJECTION_COMPLETED,
        TrialEventType.TRIAL_COMPLETED,
        TrialEventType.TRIAL_FAILED,
    }


def test_artifact_role_values():
    assert set(ArtifactRole) == {
        ArtifactRole.ITEM_PAYLOAD,
        ArtifactRole.RENDERED_PROMPT,
        ArtifactRole.INFERENCE_OUTPUT,
        ArtifactRole.METRIC_DETAILS,
        ArtifactRole.JUDGE_AUDIT,
    }


def test_issue_severity_values():
    assert set(IssueSeverity) == {IssueSeverity.ERROR, IssueSeverity.WARNING}


def test_value_hardening_enum_values():
    assert set(PValueCorrection) == {
        PValueCorrection.NONE,
        PValueCorrection.HOLM,
        PValueCorrection.BH,
    }
    assert set(DatasetSource) == {
        DatasetSource.HUGGINGFACE,
        DatasetSource.LOCAL,
        DatasetSource.MEMORY,
    }
    assert set(StorageBackend) == {
        StorageBackend.SQLITE_BLOB,
        StorageBackend.POSTGRES_BLOB,
    }
    assert set(CompressionCodec) == {
        CompressionCodec.NONE,
        CompressionCodec.ZSTD,
    }
    assert set(ResponseFormat) == {ResponseFormat.TEXT, ResponseFormat.JSON}
    assert set(SamplingKind) == {
        SamplingKind.ALL,
        SamplingKind.SUBSET,
        SamplingKind.STRATIFIED,
    }
    assert set(RecordType) == {RecordType.TRIAL, RecordType.CANDIDATE}


def test_timeline_stage_values():
    assert set(TimelineStage) == {
        TimelineStage.ITEM_LOAD,
        TimelineStage.PROMPT_RENDER,
        TimelineStage.INFERENCE,
        TimelineStage.EXTRACTION,
        TimelineStage.EVALUATION,
        TimelineStage.PROJECTION,
    }
