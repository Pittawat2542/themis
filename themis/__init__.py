"""Public package surface for Themis v6."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

from themis.api import (
    DatasetSource,
    Evaluation,
    Experiment,
    Generation,
    RunOptions,
    evaluate,
)
from themis.core.models import Case, Dataset, MetricInterpretation, MetricResult
from themis.core.results import RunResult
from themis.core.snapshot import RunSnapshot


def _resolve_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject_path.is_file():
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        return str(payload["project"]["version"])
    try:
        return version("themis-eval")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()

__all__ = [
    "Case",
    "Dataset",
    "DatasetSource",
    "Evaluation",
    "Experiment",
    "Generation",
    "MetricResult",
    "MetricInterpretation",
    "RunOptions",
    "RunResult",
    "RunSnapshot",
    "__version__",
    "evaluate",
]
