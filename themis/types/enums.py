"""Stable enums shared across persisted records, specs, and CLI surfaces."""

from enum import Enum

# Discriminator Convention (Spec 1.4):
# Enum types are reserved for status/category fields that are not used as Pydantic discriminators.
# Literal unions are used for Pydantic discriminators.


class RecordStatus(Enum):
    """Outcome state recorded for trial and candidate level records."""

    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class InferenceStatus(Enum):
    """Outcome state for a single provider inference attempt."""

    OK = "ok"
    ERROR = "error"


class ErrorWhere(Enum):
    """Pipeline stage responsible for an error record."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    INFERENCE = "inference"
    EXTRACTOR = "extractor"
    METRIC = "metric"
    STORAGE = "storage"


# Keep error codes string-valued and stable for storage, hashing, and reports.
class ErrorCode(str, Enum):
    """Stable machine-readable error code stored in artifacts and reports."""

    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PARSE_ERROR = "parse_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    METRIC_COMPUTATION = "metric_computation"
    STORAGE_WRITE = "storage_write"
    STORAGE_READ = "storage_read"
    PLUGIN_INCOMPATIBLE = "plugin_incompatible"
    MISSING_OPTIONAL_DEPENDENCY = "missing_optional_dependency"
    CIRCUIT_BREAKER = "circuit_breaker"
    ITEM_LOAD = "item_load"


class IssueSeverity(str, Enum):
    """Severity assigned to validation, compatibility, or report issues."""

    ERROR = "error"
    WARNING = "warning"


class PValueCorrection(str, Enum):
    """Multiple-comparison correction applied to p-values."""

    NONE = "none"
    HOLM = "holm"
    BH = "bh"


class DatasetSource(str, Enum):
    """Origin of dataset items loaded into an experiment."""

    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MEMORY = "memory"


class StorageBackend(str, Enum):
    """Persisted storage backend kind for a run database."""

    SQLITE_BLOB = "sqlite_blob"
    POSTGRES_BLOB = "postgres_blob"


class CompressionCodec(str, Enum):
    """Compression algorithm used for persisted large payloads."""

    NONE = "none"
    ZSTD = "zstd"


class ResponseFormat(str, Enum):
    """Structured response format requested from an engine."""

    TEXT = "text"
    JSON = "json"


class SamplingKind(str, Enum):
    """Dataset sampling strategy selected for an experiment."""

    ALL = "all"
    SUBSET = "subset"
    STRATIFIED = "stratified"


class RecordType(str, Enum):
    """Top-level persisted record family stored for a run."""

    TRIAL = "trial"
    CANDIDATE = "candidate"


class PromptRole(str, Enum):
    """Conversation role attached to prompts and tool traces."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RunStage(str, Enum):
    """High-level orchestration stage for a work item or progress snapshot."""

    GENERATION = "generation"
    TRANSFORM = "transform"
    EVALUATION = "evaluation"
