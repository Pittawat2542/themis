"""Public package surface for Themis."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

from themis.core import (
    DatasetSourceSpec,
    Experiment,
    ExecutionCheckpoint,
    InMemoryRunStore,
    PromptSpec,
    Reporter,
    RuntimeConfig,
    SessionConfig,
    RunEstimate,
    RunLineage,
    RunQuery,
    RunRecord,
    RunResult,
    RunSnapshot,
    RunStatus,
    RunStore,
    ProjectionCursor,
    RerunPlan,
    RerunSelector,
    SqliteRunStore,
    StatsEngine,
    evaluate,
    evaluate_async,
    export_evaluation_bundle,
    export_generation_bundle,
    export_parse_bundle,
    export_reduction_bundle,
    export_score_bundle,
    get_case_audit,
    get_evaluation_execution,
    get_execution_state,
    get_run_record,
    get_run_snapshot,
    get_telemetry_summary,
    import_evaluation_bundle,
    import_generation_bundle,
    import_parse_bundle,
    import_reduction_bundle,
    import_score_bundle,
    quickcheck,
    snapshot_report,
    sqlite_store,
)


def _resolve_version() -> str:
    """Resolve the installed package version with a source-tree fallback."""

    try:
        return version("themis-eval")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if pyproject_path.is_file():
            payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            return str(payload["project"]["version"])
        return "0+unknown"


__version__ = _resolve_version()

__all__ = [
    "Experiment",
    "ExecutionCheckpoint",
    "DatasetSourceSpec",
    "InMemoryRunStore",
    "PromptSpec",
    "Reporter",
    "RuntimeConfig",
    "SessionConfig",
    "RunEstimate",
    "RunLineage",
    "RunQuery",
    "RunRecord",
    "RunResult",
    "RunSnapshot",
    "RunStatus",
    "RunStore",
    "ProjectionCursor",
    "RerunPlan",
    "RerunSelector",
    "__version__",
    "SqliteRunStore",
    "StatsEngine",
    "evaluate",
    "evaluate_async",
    "export_evaluation_bundle",
    "export_generation_bundle",
    "export_parse_bundle",
    "export_reduction_bundle",
    "export_score_bundle",
    "get_case_audit",
    "get_evaluation_execution",
    "get_execution_state",
    "get_run_record",
    "get_run_snapshot",
    "get_telemetry_summary",
    "import_evaluation_bundle",
    "import_generation_bundle",
    "import_parse_bundle",
    "import_reduction_bundle",
    "import_score_bundle",
    "quickcheck",
    "snapshot_report",
    "sqlite_store",
]
