"""Helpers for working with the HuggingFaceH4/MATH-500 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator

_DATASET_NAME = "HuggingFaceH4/MATH-500"


class MathSample(BaseModel):
    unique_id: str
    problem: str
    solution: str
    answer: str
    subject: str = Field(default="unknown")
    level: int | str = Field(default=0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, value: Any) -> int | str:
        if value in (None, ""):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    @field_validator("extra", mode="before")
    @classmethod
    def _ensure_extra(cls, value: Any) -> dict[str, Any]:
        return dict(value or {})

    def to_generation_example(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload.pop("extra", None)
        payload.update(self.extra)
        return payload


def load_math500(
    *,
    split: str = "test",
    limit: int | None = None,
    subjects: Sequence[str] | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
) -> List[MathSample]:
    """Load MATH-500 samples from Hugging Face or a local directory."""

    if source not in {"huggingface", "local"}:
        raise ValueError(
            f"Unsupported source '{source}'. Expected one of: 'huggingface', 'local'."
        )

    if source == "huggingface":
        rows = _load_from_huggingface(split=split)
    else:
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided when source='local'. "
                "Pass dataset.data_dir in configs or --data-dir on the CLI."
            )
        rows = _load_from_local(Path(data_dir))

    samples: list[MathSample] = []
    selected_subjects = {s.lower() for s in subjects} if subjects else None
    for row in rows:
        if (
            selected_subjects
            and row.get("subject", "").lower() not in selected_subjects
        ):
            continue
        samples.append(_row_to_sample(row))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any]) -> MathSample:
    core_keys = {"unique_id", "problem", "solution", "answer", "subject", "level"}
    extra = {key: value for key, value in row.items() if key not in core_keys}
    data = {
        "unique_id": str(row["unique_id"]),
        "problem": row.get("problem", ""),
        "solution": row.get("solution", ""),
        "answer": row.get("answer", ""),
        "subject": row.get("subject", "unknown"),
        "level": row.get("level", 0),
        "extra": extra,
    }
    return MathSample.model_validate(data)


def _load_from_huggingface(*, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load MATH-500 from Hugging Face. Install it via `uv pip install '.[math]'`."
        ) from exc

    dataset = load_dataset(_DATASET_NAME, split=split)
    for row in dataset:
        yield dict(row)


def _load_from_local(root: Path) -> Iterator[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {root}")
    for path in root.rglob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
            row.setdefault("unique_id", path.stem)
            yield row


__all__ = ["MathSample", "load_math500"]
