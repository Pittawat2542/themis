from enum import Enum

# Discriminator Convention (Spec 1.4):
# Enum types are reserved for status/category fields that are not used as Pydantic discriminators.
# Literal unions are used for Pydantic discriminators.


class RecordStatus(Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class InferenceStatus(Enum):
    OK = "ok"
    ERROR = "error"


class ErrorWhere(Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    INFERENCE = "inference"
    EXTRACTOR = "extractor"
    METRIC = "metric"
    STORAGE = "storage"


# Keep error codes string-valued and stable for storage, hashing, and reports.
class ErrorCode(str, Enum):
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
    ERROR = "error"
    WARNING = "warning"


class PValueCorrection(str, Enum):
    NONE = "none"
    HOLM = "holm"
    BH = "bh"


class DatasetSource(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MEMORY = "memory"


class StorageBackend(str, Enum):
    SQLITE_BLOB = "sqlite_blob"
    POSTGRES_BLOB = "postgres_blob"


class CompressionCodec(str, Enum):
    NONE = "none"
    ZSTD = "zstd"


class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class SamplingKind(str, Enum):
    ALL = "all"
    SUBSET = "subset"
    STRATIFIED = "stratified"


class RecordType(str, Enum):
    TRIAL = "trial"
    CANDIDATE = "candidate"


class PromptRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RunStage(str, Enum):
    GENERATION = "generation"
    TRANSFORM = "transform"
    EVALUATION = "evaluation"
