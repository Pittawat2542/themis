"""Dataset loading helpers for the OpenAI example experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from themis.datasets import math500 as math500_dataset

from .config import DatasetConfig

_DEMO_ROWS = [
    {
        "unique_id": "demo-1",
        "problem": "Convert the point (0,3) in rectangular coordinates to polar coordinates.",
        "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
        "subject": "precalculus",
        "level": 2,
    },
    {
        "unique_id": "demo-2",
        "problem": "What is 7 + 5?",
        "answer": "12",
        "subject": "arithmetic",
        "level": 1,
    },
]


def load_dataset(config: DatasetConfig) -> List[dict[str, object]]:
    if config.kind == "demo":
        rows = [dict(row) for row in _DEMO_ROWS]
    elif config.kind == "math500_local":
        if config.data_dir is None:
            raise ValueError("data_dir must be provided for math500_local datasets")
        rows = _load_math500(
            source="local",
            limit=config.limit,
            subjects=config.subjects,
            data_dir=config.data_dir,
        )
    elif config.kind == "math500_hf":
        rows = _load_math500(
            source="huggingface", limit=config.limit, subjects=config.subjects
        )
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown dataset kind: {config.kind}")

    for row in rows:
        row.setdefault("dataset_name", config.name)
        row.setdefault("subject", row.get("subject", "unknown"))
        row.setdefault("level", row.get("level", 0))
    return rows


def _load_math500(
    *,
    source: str,
    limit: int | None,
    subjects: Iterable[str] | None,
    data_dir: Path | None = None,
):
    samples = math500_dataset.load_math500(
        source=source,
        data_dir=data_dir,
        limit=limit,
        subjects=subjects,
    )
    return [sample.to_generation_example() for sample in samples]


__all__ = ["load_dataset"]